#!/bin/sh

set -eu

repository_root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
installer=$repository_root/scripts/install.sh
temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/torch-check-installer-test.XXXXXXXX")

cleanup() {
    rm -rf -- "$temporary_directory"
}

trap cleanup 0
trap 'exit 1' HUP INT TERM

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

release_version=9.8.7
tag=v$release_version
fixtures=$temporary_directory/fixtures
mock_bin=$temporary_directory/mock-bin
mkdir -p "$fixtures" "$mock_bin"

if [ -n "${PYTHON:-}" ] && command -v "$PYTHON" >/dev/null 2>&1; then
    test_python=$PYTHON
elif command -v python3 >/dev/null 2>&1; then
    test_python=python3
elif command -v python >/dev/null 2>&1; then
    test_python=python
else
    fail "python3 or python is required to construct the PowerShell-compatible ZIP fixture"
fi

create_tar_fixture() {
    target=$1
    root=torch-check-$tag-$target
    mkdir -p "$fixtures/$root"
    printf '%s\n' '#!/bin/sh' "printf '%s\\n' 'torch-check $release_version'" >"$fixtures/$root/torch-check"
    chmod 755 "$fixtures/$root/torch-check"
    tar -C "$fixtures" -czf "$fixtures/$root.tar.gz" "$root"
}

create_zip_fixture() {
    target=$1
    root=torch-check-$tag-$target
    mkdir -p "$fixtures/$root"
    printf '%s\n' '#!/bin/sh' "printf '%s\\n' 'torch-check $release_version'" >"$fixtures/$root/torch-check.exe"
    chmod 755 "$fixtures/$root/torch-check.exe"
    "$test_python" -c 'import sys, zipfile; archive, source, member = sys.argv[1:]; handle = zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED); handle.write(source, member); handle.close()' \
        "$fixtures/$root.zip" \
        "$fixtures/$root/torch-check.exe" \
        "${root}\\torch-check.exe"
}

create_tar_fixture x86_64-unknown-linux-musl
create_tar_fixture aarch64-unknown-linux-musl
create_tar_fixture x86_64-apple-darwin
create_tar_fixture aarch64-apple-darwin
create_zip_fixture x86_64-pc-windows-msvc

: >"$fixtures/SHA256SUMS"
for archive in "$fixtures"/torch-check-*.tar.gz "$fixtures"/torch-check-*.zip; do
    if command -v sha256sum >/dev/null 2>&1; then
        checksum=$(sha256sum "$archive" | awk '{ print $1 }')
    else
        checksum=$(shasum -a 256 "$archive" | awk '{ print $1 }')
    fi
    printf '%s  %s\n' "$checksum" "${archive##*/}" >>"$fixtures/SHA256SUMS"
done

cat >"$mock_bin/uname" <<'EOF'
#!/bin/sh
case "${1:-}" in
    -s) printf '%s\n' "$MOCK_UNAME_S" ;;
    -m) printf '%s\n' "$MOCK_UNAME_M" ;;
    *) exit 1 ;;
esac
EOF

cat >"$mock_bin/curl" <<'EOF'
#!/bin/sh
set -eu
output=
url=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output | --write-out | --retry | --retry-delay | --connect-timeout | --max-time | --retry-max-time | --max-filesize | --proto | --proto-redir | --user-agent)
            if [ "$1" = --output ]; then
                output=$2
            fi
            shift 2
            ;;
        --disable | --fail | --location | --silent | --show-error | --tlsv1.2)
            shift
            ;;
        *)
            url=$1
            shift
            ;;
    esac
done

case "$url" in
    */releases/latest)
        printf '%s\n' 'https://github.com/rice8y/torch-check/releases/tag/v9.8.7'
        ;;
    */SHA256SUMS)
        cp "$MOCK_FIXTURES/SHA256SUMS" "$output"
        ;;
    */torch-check-*)
        cp "$MOCK_FIXTURES/${url##*/}" "$output"
        if [ "${MOCK_CORRUPT_DOWNLOAD:-0}" = 1 ]; then
            printf 'corrupt' >>"$output"
        fi
        ;;
    *)
        printf 'unexpected curl URL: %s\n' "$url" >&2
        exit 1
        ;;
esac
EOF

chmod 755 "$mock_bin/uname" "$mock_bin/curl"

run_success_case() {
    os=$1
    architecture=$2
    executable=$3
    use_latest=$4
    home=$temporary_directory/home-$os-$architecture
    mkdir -p "$home"

    if [ "$use_latest" = 1 ]; then
        MOCK_UNAME_S=$os MOCK_UNAME_M=$architecture MOCK_FIXTURES=$fixtures \
            HOME=$home PATH=$mock_bin:$PATH \
            sh "$installer" --quiet
    else
        MOCK_UNAME_S=$os MOCK_UNAME_M=$architecture MOCK_FIXTURES=$fixtures \
            HOME=$home PATH=$mock_bin:$PATH \
            sh "$installer" --quiet --version "$release_version"
    fi

    installed=$home/.local/bin/$executable
    [ -x "$installed" ] || fail "$os/$architecture did not install $executable"
    [ "$("$installed" --version)" = "torch-check $release_version" ] || fail "$installed reports the wrong version"
}

run_success_case Linux x86_64 torch-check 1
run_success_case Linux aarch64 torch-check 0
run_success_case Darwin x86_64 torch-check 0
run_success_case Darwin arm64 torch-check 0
run_success_case MINGW64_NT-10.0 x86_64 torch-check.exe 0

no_ulimit_home=$temporary_directory/home-no-ulimit
mkdir -p "$no_ulimit_home"
MOCK_UNAME_S=MINGW64_NT-10.0 MOCK_UNAME_M=x86_64 MOCK_FIXTURES=$fixtures \
    HOME=$no_ulimit_home PATH=$mock_bin:$PATH \
    sh -c 'ulimit() { return 1; }; installer_path=$1; shift; . "$installer_path"' \
    sh "$installer" --quiet --version "$release_version"
no_ulimit_binary=$no_ulimit_home/.local/bin/torch-check.exe
[ -x "$no_ulimit_binary" ] || fail "installer failed when ulimit -f was unavailable"
[ "$("$no_ulimit_binary" --version)" = "torch-check $release_version" ] || fail "no-ulimit install reports the wrong version"

unmodifiable_limit_home=$temporary_directory/home-unmodifiable-limit
mkdir -p "$unmodifiable_limit_home"
MOCK_UNAME_S=MINGW64_NT-10.0 MOCK_UNAME_M=x86_64 MOCK_FIXTURES=$fixtures \
    HOME=$unmodifiable_limit_home PATH=$mock_bin:$PATH \
    sh -c 'ulimit() { if [ "$#" -eq 1 ] && [ "$1" = -f ]; then printf "%s\n" unlimited; else return 1; fi; }; installer_path=$1; shift; . "$installer_path"' \
    sh "$installer" --quiet --version "$release_version"
unmodifiable_limit_binary=$unmodifiable_limit_home/.local/bin/torch-check.exe
[ -x "$unmodifiable_limit_binary" ] || fail "installer failed when ulimit -f could not be changed"

inherited_limit_home=$temporary_directory/home-inherited-limit
mkdir -p "$inherited_limit_home"
MOCK_UNAME_S=Linux MOCK_UNAME_M=x86_64 MOCK_FIXTURES=$fixtures \
    HOME=$inherited_limit_home PATH=$mock_bin:$PATH \
    sh -c 'ulimit() { if [ "$#" -eq 1 ] && [ "$1" = -f ]; then printf "%s\n" 2048; else return 1; fi; }; installer_path=$1; shift; . "$installer_path"' \
    sh "$installer" --quiet --version "$release_version"
inherited_limit_binary=$inherited_limit_home/.local/bin/torch-check
[ -x "$inherited_limit_binary" ] || fail "installer raised or rejected a stricter inherited file-size limit"

preserved_home=$temporary_directory/home-preserve
mkdir -p "$preserved_home/.local/bin"
printf '%s\n' '#!/bin/sh' "printf '%s\\n' 'preserved'" >"$preserved_home/.local/bin/torch-check"
chmod 755 "$preserved_home/.local/bin/torch-check"

if MOCK_UNAME_S=Linux MOCK_UNAME_M=x86_64 MOCK_FIXTURES=$fixtures MOCK_CORRUPT_DOWNLOAD=1 \
    HOME=$preserved_home PATH=$mock_bin:$PATH \
    sh "$installer" --quiet --version "$release_version" >"$temporary_directory/corrupt.stdout" 2>"$temporary_directory/corrupt.stderr"; then
    fail "a corrupted archive was accepted"
fi

grep -q 'checksum verification failed' "$temporary_directory/corrupt.stderr" || fail "checksum failure was not reported"
[ "$("$preserved_home/.local/bin/torch-check")" = preserved ] || fail "failed installation replaced the existing binary"

if MOCK_UNAME_S=Linux MOCK_UNAME_M=i686 MOCK_FIXTURES=$fixtures \
    HOME=$temporary_directory/unsupported PATH=$mock_bin:$PATH \
    sh "$installer" --quiet --version "$release_version" >"$temporary_directory/unsupported.stdout" 2>"$temporary_directory/unsupported.stderr"; then
    fail "an unsupported architecture was accepted"
fi

grep -q 'no Linux release binary is available' "$temporary_directory/unsupported.stderr" || fail "unsupported architecture was not explained"
sh "$installer" --help >/dev/null

printf '%s\n' 'installer tests passed'
