#!/bin/sh

set -eu

REPOSITORY_URL=https://github.com/rice8y/torch-check
PROGRAM_NAME=torch-check
MAX_ARCHIVE_BYTES=134217728
MAX_CHECKSUM_BYTES=1048576
MAX_BINARY_BYTES=134217728
MAX_BINARY_BLOCKS=262144

version=${TORCH_CHECK_VERSION:-latest}
install_dir=${TORCH_CHECK_INSTALL_DIR:-}
quiet=${TORCH_CHECK_QUIET:-0}
temporary_directory=
staged_install=

usage() {
    cat <<'EOF'
Install a torch-check binary from GitHub Releases.

Usage:
  install.sh [OPTIONS]

Options:
  --version <VERSION>      Install a release such as 0.1.0 or v0.1.0
                           (default: latest)
  --install-dir <DIR>      Install into DIR (default: $HOME/.local/bin)
  -q, --quiet              Suppress informational output
  -h, --help               Print this help

Environment:
  TORCH_CHECK_VERSION      Default value for --version
  TORCH_CHECK_INSTALL_DIR  Default value for --install-dir
  TORCH_CHECK_QUIET        Set to 1 to suppress informational output

The archive and SHA256SUMS file are downloaded over HTTPS. Verification cannot
be disabled. The installer does not invoke sudo or modify shell startup files.
EOF
}

say() {
    if [ "$quiet" != 1 ]; then
        printf '%s\n' "$*"
    fi
}

warn() {
    printf 'warning: %s\n' "$*" >&2
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [ -n "$staged_install" ]; then
        rm -f -- "$staged_install"
    fi
    if [ -n "$temporary_directory" ]; then
        rm -rf -- "$temporary_directory"
    fi
}

trap cleanup 0
trap 'exit 1' HUP INT TERM

require_argument() {
    option=$1
    remaining=$2
    if [ "$remaining" -lt 2 ]; then
        die "$option requires a value"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            require_argument "$1" "$#"
            version=$2
            shift 2
            ;;
        --version=*)
            version=${1#*=}
            shift
            ;;
        --install-dir)
            require_argument "$1" "$#"
            install_dir=$2
            shift 2
            ;;
        --install-dir=*)
            install_dir=${1#*=}
            shift
            ;;
        -q | --quiet)
            quiet=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            if [ "$#" -ne 0 ]; then
                die "unexpected positional argument: $1"
            fi
            ;;
        -*)
            die "unknown option: $1 (try --help)"
            ;;
        *)
            die "unexpected positional argument: $1"
            ;;
    esac
done

case "$quiet" in
    0 | 1) ;;
    *) die "TORCH_CHECK_QUIET must be 0 or 1" ;;
esac

if [ -z "$install_dir" ]; then
    if [ -z "${HOME:-}" ]; then
        die "HOME is unset; use --install-dir or TORCH_CHECK_INSTALL_DIR"
    fi
    install_dir=$HOME/.local/bin
fi

case "$install_dir" in
    /*) ;;
    *) die "install directory must be an absolute path: $install_dir" ;;
esac

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v mktemp >/dev/null 2>&1 || die "mktemp is required"
command -v awk >/dev/null 2>&1 || die "awk is required"
command -v grep >/dev/null 2>&1 || die "grep is required"
command -v sed >/dev/null 2>&1 || die "sed is required"
command -v tr >/dev/null 2>&1 || die "tr is required"
command -v uname >/dev/null 2>&1 || die "uname is required"
command -v wc >/dev/null 2>&1 || die "wc is required"

os=$(uname -s 2>/dev/null) || die "could not detect the operating system"
architecture=$(uname -m 2>/dev/null) || die "could not detect the architecture"

case "$os" in
    Linux)
        archive_format=tar.gz
        executable=$PROGRAM_NAME
        case "$architecture" in
            x86_64 | amd64) target=x86_64-unknown-linux-musl ;;
            aarch64 | arm64) target=aarch64-unknown-linux-musl ;;
            *) die "no Linux release binary is available for architecture $architecture" ;;
        esac
        ;;
    Darwin)
        archive_format=tar.gz
        executable=$PROGRAM_NAME
        case "$architecture" in
            x86_64 | amd64) target=x86_64-apple-darwin ;;
            aarch64 | arm64) target=aarch64-apple-darwin ;;
            *) die "no macOS release binary is available for architecture $architecture" ;;
        esac
        ;;
    MINGW* | MSYS* | CYGWIN*)
        archive_format=zip
        executable=$PROGRAM_NAME.exe
        case "$architecture" in
            x86_64 | amd64) target=x86_64-pc-windows-msvc ;;
            *) die "no Windows release binary is available for architecture $architecture" ;;
        esac
        ;;
    *)
        die "unsupported operating system: $os"
        ;;
esac

detect_linux_libc() {
    if command -v getconf >/dev/null 2>&1; then
        libc_description=$(getconf GNU_LIBC_VERSION 2>/dev/null || true)
        case "$libc_description" in
            glibc\ *) printf '%s\n' glibc; return ;;
        esac
    fi

    if command -v ldd >/dev/null 2>&1; then
        libc_description=$(ldd --version 2>&1 || true)
        case "$libc_description" in
            *musl* | *MUSL*) printf '%s\n' musl; return ;;
            *GLIBC* | *glibc* | *GNU\ libc*) printf '%s\n' glibc; return ;;
        esac
    fi

    for loader in /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1; do
        if [ -e "$loader" ]; then
            printf '%s\n' musl
            return
        fi
    done

    printf '%s\n' unknown
}

if [ "$os" = Linux ]; then
    host_libc=$(detect_linux_libc)
    case "$host_libc:$architecture" in
        glibc:x86_64 | glibc:amd64) ;;
        *) warn "this binary can inspect $os/$architecture/$host_libc, but recommendations are currently supported only on Linux x86_64 with glibc" ;;
    esac
fi

normalize_version() {
    requested=$1
    case "$requested" in
        v*) requested=${requested#v} ;;
    esac

    if ! printf '%s\n' "$requested" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        return 1
    fi

    printf 'v%s\n' "$requested"
}

curl_download() {
    source_url=$1
    destination=$2
    maximum_bytes=$3
    maximum_blocks=$((maximum_bytes / 512 + 1))
    (
        ulimit -f "$maximum_blocks" || exit 1
        curl --disable \
            --fail \
            --location \
            --silent \
            --show-error \
            --retry 3 \
            --retry-delay 1 \
            --connect-timeout 15 \
            --max-time 300 \
            --retry-max-time 300 \
            --max-filesize "$maximum_bytes" \
            --proto '=https' \
            --proto-redir '=https' \
            --tlsv1.2 \
            --user-agent 'torch-check-installer' \
            --output "$destination" \
            "$source_url"
    )
}

if [ "$version" = latest ]; then
    latest_url=$(curl --disable \
        --fail \
        --location \
        --silent \
        --show-error \
        --retry 3 \
        --retry-delay 1 \
        --connect-timeout 15 \
        --max-time 60 \
        --retry-max-time 60 \
        --proto '=https' \
        --proto-redir '=https' \
        --tlsv1.2 \
        --user-agent 'torch-check-installer' \
        --output /dev/null \
        --write-out '%{url_effective}' \
        "$REPOSITORY_URL/releases/latest") || die "could not resolve the latest release"
    latest_url=${latest_url%%\?*}
    latest_url=${latest_url%/}
    version=${latest_url##*/}
fi

tag=$(normalize_version "$version") || die "invalid release version: $version"
release_version=${tag#v}
archive_name=$PROGRAM_NAME-$tag-$target.$archive_format
archive_root=$PROGRAM_NAME-$tag-$target
release_url=$REPOSITORY_URL/releases/download/$tag

temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/torch-check-install.XXXXXXXX") || die "could not create a temporary directory"
archive_path=$temporary_directory/$archive_name
checksums_path=$temporary_directory/SHA256SUMS
extracted_path=$temporary_directory/$executable

say "Downloading $PROGRAM_NAME $release_version for $target"
curl_download "$release_url/$archive_name" "$archive_path" "$MAX_ARCHIVE_BYTES" || die "could not download $archive_name"
curl_download "$release_url/SHA256SUMS" "$checksums_path" "$MAX_CHECKSUM_BYTES" || die "could not download SHA256SUMS"

file_size() {
    measured_size=$(wc -c <"$1") || return 1
    measured_size=$(printf '%s' "$measured_size" | tr -d '[:space:]')
    case "$measured_size" in
        '' | *[!0-9]*) return 1 ;;
    esac
    printf '%s\n' "$measured_size"
}

require_size_at_most() {
    bounded_file=$1
    maximum_size=$2
    description=$3
    bounded_size=$(file_size "$bounded_file") || die "could not measure $description"
    if [ "$bounded_size" -gt "$maximum_size" ]; then
        die "$description exceeds the $maximum_size-byte safety limit"
    fi
}

require_size_at_most "$archive_path" "$MAX_ARCHIVE_BYTES" "$archive_name"
require_size_at_most "$checksums_path" "$MAX_CHECKSUM_BYTES" SHA256SUMS

expected_checksum=$(awk -v file="$archive_name" '
    {
        name = $2
        sub(/^\*/, "", name)
        if (name == file) {
            count += 1
            checksum = $1
        }
    }
    END {
        if (count != 1) {
            exit 1
        }
        print checksum
    }
' "$checksums_path") || die "SHA256SUMS does not contain exactly one entry for $archive_name"

sha256_file() {
    file=$1
    if command -v sha256sum >/dev/null 2>&1; then
        digest_output=$(sha256sum "$file") || return 1
        printf '%s\n' "$digest_output" | awk '{ print $1 }'
    elif command -v shasum >/dev/null 2>&1; then
        digest_output=$(shasum -a 256 "$file") || return 1
        printf '%s\n' "$digest_output" | awk '{ print $1 }'
    elif command -v openssl >/dev/null 2>&1; then
        digest_output=$(openssl dgst -sha256 "$file") || return 1
        printf '%s\n' "$digest_output" | awk '{ print $NF }'
    else
        die "sha256sum, shasum, or openssl is required to verify the download"
    fi
}

actual_checksum=$(sha256_file "$archive_path") || die "could not calculate the archive checksum"
for checksum in "$expected_checksum" "$actual_checksum"; do
    case "$checksum" in
        *[!0-9A-Fa-f]* | '') die "invalid SHA-256 digest" ;;
    esac
    if [ "${#checksum}" -ne 64 ]; then
        die "invalid SHA-256 digest"
    fi
done

expected_checksum=$(printf '%s' "$expected_checksum" | tr 'A-F' 'a-f')
actual_checksum=$(printf '%s' "$actual_checksum" | tr 'A-F' 'a-f')
if [ "$expected_checksum" != "$actual_checksum" ]; then
    die "checksum verification failed for $archive_name"
fi
say "Verified SHA-256 checksum"

archive_member=$archive_root/$executable
case "$archive_format" in
    tar.gz)
        command -v tar >/dev/null 2>&1 || die "tar is required"
        member_count=$(tar -tzf "$archive_path" | awk -v member="$archive_member" '$0 == member { count += 1 } END { print count + 0 }') || die "could not inspect $archive_name"
        if [ "$member_count" -ne 1 ]; then
            die "$archive_name does not contain exactly one $archive_member"
        fi
        (
            ulimit -f "$MAX_BINARY_BLOCKS" || exit 1
            tar -xzOf "$archive_path" "$archive_member" >"$extracted_path"
        ) || die "could not safely extract $archive_member"
        ;;
    zip)
        command -v unzip >/dev/null 2>&1 || die "unzip is required"
        zip_member=$(unzip -Z1 "$archive_path" | awk -v member="$archive_member" '
            {
                normalized = $0
                gsub(/\\/, "/", normalized)
                if (normalized == member) {
                    count += 1
                    original = $0
                }
            }
            END {
                if (count != 1) {
                    exit 1
                }
                print original
            }
        ') || die "$archive_name does not contain exactly one $archive_member"
        if [ -z "$zip_member" ]; then
            die "$archive_name does not contain exactly one $archive_member"
        fi
        zip_pattern=$(printf '%s' "$zip_member" | sed 's,\\,\\\\,g')
        (
            ulimit -f "$MAX_BINARY_BLOCKS" || exit 1
            unzip -p "$archive_path" "$zip_pattern" >"$extracted_path"
        ) || die "could not safely extract $archive_member"
        ;;
esac

require_size_at_most "$extracted_path" "$MAX_BINARY_BYTES" "$executable"

chmod 755 "$extracted_path" || die "could not mark the downloaded binary executable"
reported_version=$("$extracted_path" --version 2>/dev/null) || die "the downloaded binary could not be executed on this host"
if [ "$reported_version" != "$PROGRAM_NAME $release_version" ]; then
    die "downloaded binary reported an unexpected version: $reported_version"
fi

mkdir -p "$install_dir" || die "could not create install directory: $install_dir"
if [ -d "$install_dir/$executable" ]; then
    die "install destination is a directory: $install_dir/$executable"
fi

staged_install=$(mktemp "$install_dir/.torch-check-install.XXXXXXXX") || die "install directory is not writable: $install_dir"
cp "$extracted_path" "$staged_install" || die "could not stage the binary in $install_dir"
chmod 755 "$staged_install" || die "could not set executable permissions"
mv -f "$staged_install" "$install_dir/$executable" || die "could not install to $install_dir/$executable"
staged_install=

say "Installed $reported_version to $install_dir/$executable"
case ":${PATH:-}:" in
    *:"$install_dir":*) ;;
    *) warn "$install_dir is not on PATH; add it to your shell configuration" ;;
esac
