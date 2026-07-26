import base64
import binascii
import posixpath
import re
import shlex
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Agent Tool Guardrail API")


# -------------------------------------------------------------------
# Policy configuration
# -------------------------------------------------------------------

WORKING_DIRECTORY = "/home/agent/workspace"
HOME_DIRECTORY = "/home/agent"
RESTRICTED_FILE = "/home/agent/.pgpass"
ALLOWED_WRITE_DIRECTORY = "/data/agent/outbox"

ALLOWED_HTTP_HOSTS = {
    "api.github.com",
    "objects.githubusercontent.com",
}


# -------------------------------------------------------------------
# Request model
# -------------------------------------------------------------------

class ToolRequest(BaseModel):
    tool: str

    # Used by the bash tool
    command: Optional[str] = None

    # Used by the write_file tool
    path: Optional[str] = None
    content: Optional[str] = None

    # Used by the http_request tool
    method: Optional[str] = None
    url: Optional[str] = None


# -------------------------------------------------------------------
# Response helpers
# -------------------------------------------------------------------

def allow(reason: str) -> dict:
    return {
        "decision": "allow",
        "reason": reason,
    }


def block(reason: str) -> dict:
    return {
        "decision": "block",
        "reason": reason,
    }


# -------------------------------------------------------------------
# Path handling
# -------------------------------------------------------------------

def expand_known_variables(value: str) -> str:
    """
    Expand only the environment variables defined by the question.
    We deliberately avoid using the actual server environment.
    """

    expanded = value

    replacements = {
        "${HOME}": HOME_DIRECTORY,
        "$HOME": HOME_DIRECTORY,
        "${PWD}": WORKING_DIRECTORY,
        "$PWD": WORKING_DIRECTORY,
    }

    for original, replacement in replacements.items():
        expanded = expanded.replace(original, replacement)

    # Expand a standalone ~ or ~/path as /home/agent
    expanded = re.sub(
        r"(?<![A-Za-z0-9_])~(?=/|$)",
        HOME_DIRECTORY,
        expanded,
    )

    return expanded


def clean_path_token(value: str) -> str:
    """Remove common shell punctuation surrounding a path."""

    return value.strip(
        " \t\r\n"
        "'\""
        "`"
        "()[]{}"
        ",;"
    )


def normalize_agent_path(path: str) -> str:
    """
    Convert a possibly relative path into a normalized Linux path.

    Relative paths are resolved from /home/agent/workspace.
    """

    expanded = expand_known_variables(path)
    cleaned = clean_path_token(expanded)

    # Handle file:// paths
    if cleaned.startswith("file://"):
        cleaned = cleaned[7:]

    if cleaned.startswith("/"):
        return posixpath.normpath(cleaned)

    return posixpath.normpath(
        posixpath.join(WORKING_DIRECTORY, cleaned)
    )


def is_restricted_path(path: str) -> bool:
    try:
        return normalize_agent_path(path) == RESTRICTED_FILE
    except Exception:
        return False


def is_inside_allowed_write_directory(path: str) -> bool:
    try:
        normalized = normalize_agent_path(path)

        return (
            normalized == ALLOWED_WRITE_DIRECTORY
            or normalized.startswith(ALLOWED_WRITE_DIRECTORY + "/")
        )
    except Exception:
        return False


# -------------------------------------------------------------------
# Base64 handling
# -------------------------------------------------------------------

BASE64_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/])"
    r"([A-Za-z0-9+/]{12,}={0,2})"
    r"(?![A-Za-z0-9+/])"
)


def decode_base64_strings(command: str) -> list[str]:
    """
    Find Base64-looking strings inside a command and decode them.

    This detects commands such as:

    echo Y2F0IC9ob21lL2FnZW50Ly5wZ3Bhc3M= | base64 -d | bash
    """

    decoded_commands: list[str] = []

    for match in BASE64_PATTERN.findall(command):
        try:
            padded = match + "=" * (-len(match) % 4)
            decoded_bytes = base64.b64decode(
                padded,
                validate=True,
            )

            decoded_text = decoded_bytes.decode("utf-8")

            # Ignore binary data
            if decoded_text and all(
                character.isprintable()
                or character in "\r\n\t"
                for character in decoded_text
            ):
                decoded_commands.append(decoded_text)

        except (
            binascii.Error,
            UnicodeDecodeError,
            ValueError,
        ):
            continue

    return decoded_commands


def get_command_variants(command: str) -> list[str]:
    """
    Return the original command plus decoded nested Base64 commands.
    """

    variants = [command]
    queue = [command]
    seen = {command}

    # Two levels are enough for ordinary nested wrappers
    for _ in range(2):
        next_queue: list[str] = []

        for current in queue:
            for decoded in decode_base64_strings(current):
                if decoded not in seen:
                    seen.add(decoded)
                    variants.append(decoded)
                    next_queue.append(decoded)

        queue = next_queue

        if not queue:
            break

    return variants


# -------------------------------------------------------------------
# Bash inspection
# -------------------------------------------------------------------

def extract_possible_paths(command: str) -> set[str]:
    """Extract path-like values from a shell command."""

    expanded = expand_known_variables(command)
    candidates: set[str] = set()

    # Direct absolute, relative and home-based paths
    path_pattern = re.compile(
        r"(?:"
        r"/[^\s|;&<>]+"
        r"|"
        r"(?:\.\.?/)[^\s|;&<>]+"
        r"|"
        r"~/[^\s|;&<>]+"
        r")"
    )

    for match in path_pattern.findall(expanded):
        candidates.add(clean_path_token(match))

    # Shell parsing handles quoted and concatenated arguments
    try:
        tokens = shlex.split(expanded, posix=True)

        for token in tokens:
            cleaned = clean_path_token(token)

            if (
                "/" in cleaned
                or cleaned.startswith(".")
                or cleaned.startswith("~")
                or ".pgpass" in cleaned
            ):
                candidates.add(cleaned)

    except ValueError:
        # Malformed shell command: continue using regex results
        pass

    return candidates


def command_references_restricted_file(command: str) -> bool:
    for variant in get_command_variants(command):
        expanded = expand_known_variables(variant)

        # Fast check for a direct absolute reference
        if RESTRICTED_FILE in expanded:
            return True

        for candidate in extract_possible_paths(variant):
            if is_restricted_path(candidate):
                return True

    return False


def extract_redirection_targets(command: str) -> list[str]:
    """
    Extract destinations from common output redirections.

    Examples:
        echo test > /tmp/file
        echo test >> /data/agent/outbox/file.txt
    """

    expanded = expand_known_variables(command)

    pattern = re.compile(
        r"(?:^|[\s])"
        r"(?:>|>>|1>|1>>|2>|2>>|&>)"
        r"\s*"
        r"([^\s|;&]+)"
    )

    return [
        clean_path_token(target)
        for target in pattern.findall(expanded)
    ]


def find_common_bash_write_targets(command: str) -> list[str]:
    """
    Detect destinations for common shell commands that create or modify
    files.
    """

    write_targets: list[str] = []

    try:
        tokens = shlex.split(
            expand_known_variables(command),
            posix=True,
        )
    except ValueError:
        return write_targets

    if not tokens:
        return write_targets

    # Divide commands on simple shell separators
    segments: list[list[str]] = []
    current: list[str] = []

    for token in tokens:
        if token in {";", "&&", "||", "|"}:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)

    if current:
        segments.append(current)

    for segment in segments:
        if not segment:
            continue

        command_name = posixpath.basename(segment[0])

        arguments = [
            argument
            for argument in segment[1:]
            if not argument.startswith("-")
        ]

        if command_name in {"touch", "mkdir", "truncate"}:
            write_targets.extend(arguments)

        elif command_name in {"cp", "mv", "install"}:
            if len(arguments) >= 2:
                # Last argument is the destination
                write_targets.append(arguments[-1])

        elif command_name == "tee":
            write_targets.extend(arguments)

    return write_targets


def bash_writes_outside_allowed_directory(command: str) -> bool:
    for variant in get_command_variants(command):
        targets = extract_redirection_targets(variant)
        targets.extend(find_common_bash_write_targets(variant))

        for target in targets:
            if not is_inside_allowed_write_directory(target):
                return True

    return False


def check_bash(command: Optional[str]) -> dict:
    if not command or not command.strip():
        return block("A non-empty bash command is required.")

    if command_references_restricted_file(command):
        return block(
            "Access to /home/agent/.pgpass is never permitted."
        )

    if bash_writes_outside_allowed_directory(command):
        return block(
            "Bash writes are permitted only inside /data/agent/outbox."
        )

    return allow(
        "The bash command does not violate the restricted-file or "
        "write-directory policy."
    )


# -------------------------------------------------------------------
# write_file inspection
# -------------------------------------------------------------------

def check_write_file(path: Optional[str]) -> dict:
    if not path or not path.strip():
        return block("A file path is required.")

    if not is_inside_allowed_write_directory(path):
        return block(
            "Files may be written only inside /data/agent/outbox."
        )

    return allow(
        "The destination is inside /data/agent/outbox."
    )


# -------------------------------------------------------------------
# HTTP inspection
# -------------------------------------------------------------------

def check_http_request(url: Optional[str]) -> dict:
    if not url or not url.strip():
        return block("A valid HTTP URL is required.")

    try:
        parsed = urlparse(url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return block(
                "Only HTTP and HTTPS URLs are permitted."
            )

        hostname = parsed.hostname

        if hostname is None:
            return block("The URL does not contain a valid hostname.")

        # Hostnames are case-insensitive.
        # A final dot represents the same DNS hostname.
        normalized_hostname = hostname.lower().rstrip(".")

        if normalized_hostname not in ALLOWED_HTTP_HOSTS:
            return block(
                "Outbound HTTP requests are allowed only to the exact "
                "approved GitHub hosts."
            )

        return allow(
            "The request hostname exactly matches an approved host."
        )

    except (TypeError, ValueError):
        return block("The supplied URL is invalid.")


# -------------------------------------------------------------------
# Main endpoint
# -------------------------------------------------------------------

@app.post("/check")
def guardrail_check(request: ToolRequest):
    tool = request.tool.strip().lower()

    if tool == "bash":
        return check_bash(request.command)

    if tool == "write_file":
        return check_write_file(request.path)

    if tool == "http_request":
        return check_http_request(request.url)

    return block(
        "This tool is not supported by the guardrail policy."
    )


@app.get("/")
def root():
    return {
        "message": "Guardrail API is running",
        "endpoint": "/check",
    }