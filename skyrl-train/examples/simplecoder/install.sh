#!/bin/sh
# GNU Guix --- Functional package management for GNU
# Copyright © 2017 sharlatan <sharlatanus@gmail.com>
# Copyright © 2018 Ricardo Wurmus <rekado@elephly.net>
# Copyright © 2018 Efraim Flashner <efraim@flashner.co.il>
# Copyright © 2019–2020, 2022 Tobias Geerinckx-Rice <me@tobias.gr>
# Copyright © 2020 Morgan Smith <Morgan.J.Smith@outlook.com>
# Copyright © 2020 Simon Tournier <zimon.toutoune@gmail.com>
# Copyright © 2020 Daniel Brooks <db48x@db48x.net>
# Copyright © 2021 Jakub Kądziołka <kuba@kadziolka.net>
# Copyright © 2021 Chris Marusich <cmmarusich@gmail.com>
# Copyright © 2021, 2022, 2023 Maxim Cournoyer <maxim.cournoyer@gmail.com>
# Copyright © 2022 Prafulla Giri <prafulla.giri@protonmail.com>
# Copyright © 2023 Andrew Tropin <andrew@trop.in>
# Copyright © 2020 David A. Redick <david.a.redick@gmail.com>
# Copyright © 2024 Janneke Nieuwenhuizen <janneke@gnu.org>
# Copyright © 2024 Tomas Volf <~@wolfsden.cz>
# Copyright © 2024 Richard Sent <richard@freakingpenguin.com>
# Copyright © 2025 Denis 'GNUtoo' Carikli <GNUtoo@cyberdimension.org>
#
# This file is part of GNU Guix.
#
# GNU Guix is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# GNU Guix is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GNU Guix.  If not, see <http://www.gnu.org/licenses/>.

# We require Bash but for portability we'd rather not use /bin/bash or
# /usr/bin/env in the shebang, hence this hack.

# Environment variables
#
# GUIX_BINARY_FILE_NAME
#
# Can be used to override the automatic download mechanism and point
# to a local Guix binary archive filename like
# "/tmp/guix-binary-1.4.0rc2.armhf-linux.tar.xz"
#
# GUIX_ALLOW_OVERWRITE
#
# Instead of aborting to avoid overwriting a previous installations,
# allow copying over /var/guix or /gnu.  This can be useful when the
# installation required the user to extract Guix packs under /gnu to
# satisfy its dependencies.

# shellcheck shell=bash

# shellcheck disable=2268 # try to support vintage shells
if [ "x$BASH_VERSION" = "x" ]; then
    exec bash "$0" "$@"
fi

set -eo pipefail

[ "$UID" -eq 0 ] || { echo "This script must be run as root."; exit 1; }

REQUIRE=(
    "dirname"
    "readlink"
    "wget"
    "gpg"
    "grep"
    "sed"
    "sort"
    "getent"
    "mktemp"
    "rm"
    "chmod"
    "uname"
    "groupadd"
    "groupdel"
    "useradd"
    "userdel"
    "tail"
    "tr"
    "xz"
)

# Add variables using form FOO_INIT_REQUIRE when init system FOO dependencies
# should be checked.
# shellcheck disable=2034 # interpolated by add_init_sys_require
SYSV_INIT_REQUIRE=(
    "daemonize"
)

PAS=$'[ \033[32;1mPASS\033[0m ] '
ERR=$'[ \033[31;1mFAIL\033[0m ] '
WAR=$'[ \033[33;1mWARN\033[0m ] '
INF="[ INFO ] "

DEBUG=0
GNU_URL="https://ftp.gnu.org/gnu/guix/"
#GNU_URL="https://alpha.gnu.org/gnu/guix/"

# The following associative array holds set of GPG keys used to sign the
# releases, keyed by their corresponding Savannah user ID.
declare -A GPG_SIGNING_KEYS
GPG_SIGNING_KEYS[15145]=3CE464558A84FDC69DB40CFB090B11993D9AEBB5  # ludo
GPG_SIGNING_KEYS[127547]=27D586A4F8900854329FF09F1260E46482E63562 # maxim

# ------------------------------------------------------------------------------
#+UTILITIES

_err()
{ # All errors go to stderr.
    printf "[%s]: ${ERR}%s\n" "$(date +%s.%3N)" "$1"
}

_msg()
{ # Default message to stdout.
    printf "[%s]: %s\n" "$(date +%s.%3N)" "$1"
}

_msg_pass()
{
    _msg "$PAS$1"
}

_msg_warn()
{
    _msg "$WAR$1"
}

_msg_info()
{
    _msg "$INF$1"
}

_debug()
{
    if [ "${DEBUG}" = '1' ]; then
        printf "[%s]: %s\n" "$(date +%s.%3N)" "$1"
    fi
}

die()
{
    _err "$*"
    exit 1
}

# Return true if user answered yes, false otherwise.  The prompt is
# yes-biased, that is, when the user simply enter newline, it is equivalent to
# answering "yes".
# $1: The prompt question.
prompt_yes_no() {
    local -l yn
    read -rp "$1 [Y/n]" yn
    [[ ! $yn || $yn = y || $yn = yes ]] || return 1
}

chk_require()
{ # Check that every required command is available.
    declare -a warn
    local c

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    for c in "$@"; do
        command -v "$c" &>/dev/null || warn+=("$c")
    done

    [ "${#warn}" -ne 0 ] && die "Missing commands: ${warn[*]}."

    _msg_pass "verification of required commands completed"
}

add_init_sys_require()
{ # Add the elements of FOO_INIT_SYS to REQUIRE

    # Convert the init system command name captured in INIT_SYS in
    # chk_init_sys to uppercase, with hyphens replaced by underscores.
    local init_require=${INIT_SYS^^}_REQUIRE[@]
    init_require=${init_require//-/_}

    if [[ -n "$init_require" ]]; then
        # Have to add piecemeal because ${!foo[@]} performs direct array key
        # expansion, not indirect plain array expansion.
        for r in "${!init_require}"; do
            REQUIRE+=("$r")
        done
    fi
}

chk_gpg_keyring()
{ # Check whether the Guix release signing public key is present.
    _debug "--- [ ${FUNCNAME[0]} ] ---"
    local user_id
    local gpg_key_id
    local exit_flag

    for user_id in "${!GPG_SIGNING_KEYS[@]}"; do
        gpg_key_id=${GPG_SIGNING_KEYS[$user_id]}
        # Without --dry-run this command will create a ~/.gnupg owned by root on
        # systems where gpg has never been used, causing errors and confusion.
        if gpg --dry-run --list-keys "$gpg_key_id" >/dev/null 2>&1; then
            continue
        fi
        if prompt_yes_no "${INF}The following OpenPGP public key is \
required to verify the Guix binary signature: $gpg_key_id.
Would you like me to fetch it for you?"; then
            # Use a reasonable time-out here so users don't report silent
            # ‘freezes’ when Savannah goes out to lunch, as has happened.
            if wget "https://sv.gnu.org/people/viewgpg.php?user_id=$user_id" \
                    --timeout=30 --no-verbose -O- | gpg --import -; then
                continue
            fi
        fi
	# If we reach this point, the key is (still) missing.  Report further
	# missing keys, if any, but then abort the installation.
        _err "Missing OpenPGP public key ($gpg_key_id).
Fetch it with this command:

  wget \"https://sv.gnu.org/people/viewgpg.php?user_id=$user_id\" -O - | \
sudo -i gpg --import -"
        exit_flag=yes
    done
    if [ "$exit_flag" = yes ]; then
        exit 1
    fi
}

chk_term()
{ # Check for ANSI terminal for color printing.
    if [ -t 2 ]; then
        if [ "${TERM+set}" = 'set' ]; then
            case "$TERM" in
                xterm*|rxvt*|urxvt*|linux*|vt*|eterm*|screen*)
                    ;;
                *)
                    ERR="[ FAIL ] "
                    PAS="[ PASS ] "
                    WAR="[ WARN ] "
                    ;;
            esac
        fi
    fi
}

chk_init_sys()
{ # Return init system type name.
    if [[ $(/sbin/init --version 2>/dev/null) =~ upstart ]]; then
        _msg_info "init system is: upstart"
        INIT_SYS="upstart"
        return 0
    elif [[ $(systemctl 2>/dev/null) =~ -\.mount ]]; then
        _msg_info "init system is: systemd"
        INIT_SYS="systemd"
        return 0
    elif [[ -f /etc/init.d/cron && ! -h /etc/init.d/cron ]]; then
        _msg_info "init system is: sysv-init"
        INIT_SYS="sysv-init"
        return 0
    elif [[ $(openrc --version 2>/dev/null) =~ \(OpenRC ]]; then
        _msg_info "init system is: OpenRC"
        INIT_SYS="openrc"
        return 0
    else
        INIT_SYS="NA"
        _err "Init system could not be detected."
    fi
}

chk_sys_arch()
{ # Check for operating system and architecture type.
    local os
    local arch

    os="$(uname -s)"
    arch="$(uname -m)"

    case "$arch" in
        i386 | i486 | i686 | i786 | x86)
            local arch=i686
            ;;
        x86_64 | x86-64 | x64 | amd64)
            local arch=x86_64
            ;;
        aarch64)
            local arch=aarch64
            ;;
        armv7l)
            local arch=armhf
            ;;
        ppc64le | powerpc64le)
            local arch=powerpc64le
            ;;
        *)
            die "Unsupported CPU type: ${arch}"
    esac

    case "$os" in
        Linux | linux)
            local os=linux
            ;;
        *)
            die "Your operation system (${os}) is not supported."
    esac

    ARCH_OS="${arch}-${os}"
}

chk_sys_nscd()
{ # Check if nscd is up and suggest to start it or install it
    if [ "$(type -P pidof)" ]; then
        if [ ! "$(pidof nscd)" ]; then
            _msg_warn "We recommend installing and/or starting your distribution 'nscd' service"
            _msg_warn "Please read 'info guix \"Application Setup\"' about \"Name Service Switch\""
        fi
    else
        _msg_info "We cannot determine if your distribution 'nscd' service is running"
        _msg_info "Please read 'info guix \"Application Setup\"' about \"Name Service Switch\""
    fi
}

chk_existing()
{ # Avoid clobbering existing installations.
    _debug "--- [ ${FUNCNAME[0]} ] ---"

    if [[ -e /var/guix && -e /gnu ]]; then
        if [ -n "$GUIX_ALLOW_OVERWRITE" ]; then
            _msg_warn "Overwriting existing installation!"
        else
            die "A previous Guix installation was found.  Refusing to overwrite."
        fi
    fi
}

# Configure substitute discovery according to user's preferences.
# $1 is the installed service file to edit.
configure_substitute_discovery() {
    if grep -q -- '--discover=no' "$1" && \
            prompt_yes_no "Would you like the Guix daemon to automatically \
discover substitute servers on the local network?"; then
        sed -i 's/--discover=no/--discover=yes/' "$1"
    fi
}

# ------------------------------------------------------------------------------
#+MAIN

guix_get_bin_list()
{ # Scan GNU archive and save list of binaries
    local gnu_url="$1"
    local -a bin_ver_ls
    local latest_ver
    local default_ver

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    # Filter only version and architecture
    bin_ver_ls=("$(wget "$gnu_url" --no-verbose -O- \
        | sed -n -e 's/.*guix-binary-\([0-9.]*[a-z0-9]*\)\..*.tar.xz.*/\1/p' \
        | sort -Vu)")

    latest_ver="$(echo "${bin_ver_ls[0]}" \
                       | grep -oE "([0-9]{1,2}\.){2}[0-9]{1,2}[a-z0-9]*" \
                       | tail -n1)"

    default_ver="guix-binary-${latest_ver}.${ARCH_OS}"

    if [[ "${#bin_ver_ls}" -ne "0" ]]; then
        _msg_pass "Release for your system: ${default_ver}"
    else
        die "Could not obtain list of Guix releases."
    fi

    # Use default to download according to the list and local ARCH_OS.
    BIN_VER="${default_ver}"
}

guix_get_bin()
{ # Download and verify binary package.
    local url="$1"
    local bin_ver="$2"
    local dl_path="$3"
    local wget_args=()

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    _msg_info "Downloading Guix release archive"

    wget --help | grep -q '\--show-progress' \
        && wget_args=("--no-verbose" "--show-progress")

    if wget "${wget_args[@]}" -P "$dl_path" \
            "${url}/${bin_ver}.tar.xz" "${url}/${bin_ver}.tar.xz.sig"; then
        _msg_pass "download completed."
    else
        die "could not download ${url}/${bin_ver}.tar.xz."
    fi

    pushd "${dl_path}" >/dev/null
    if gpg --verify "${bin_ver}.tar.xz.sig" >/dev/null 2>&1; then
        _msg_pass "Signature is valid."
        popd >/dev/null
    else
        die "could not verify the signature."
    fi
}

sys_create_store()
{ # Unpack and install /gnu/store and /var/guix
    local pkg="$1"
    local tmp_path="$2"

    cd "$tmp_path"
    _msg_info "Installing /var/guix and /gnu..."
    # Strip (skip) the leading ‘.’ component, which fails on read-only ‘/’.
    #
    # TODO: Eventually extract with ‘--owner=guix-daemon’ when installing
    # and unprivileged guix-daemon service; for now, this script may install
    # from both an old release that does not support unprivileged guix-daemon
    # and a new release that does, so ‘chown -R’ later if needed.
    tar --extract --strip-components=1 --file "$pkg" -C /

    _msg_info "Linking the root user's profile"
    mkdir -p ~root/.config/guix
    ln -sf /var/guix/profiles/per-user/root/current-guix \
       ~root/.config/guix/current

    GUIX_PROFILE=~root/.config/guix/current
    # The profile just prepends to search paths, which is not needed for
    # effective linting.
    # shellcheck disable=SC1091
    source "${GUIX_PROFILE}/etc/profile"
    _msg_pass "activated root profile at ${GUIX_PROFILE}"
}

sys_delete_store()
{
    _msg_info "removing /var/guix"
    rm -rf /var/guix

    _msg_info "removing /gnu"
    rm -rf /gnu

    _msg_info "removing ~root/.config/guix"
    rm -rf ~root/.config/guix
}

create_account()
{
    local user="$1"
    local group="$2"
    local supplementary_groups="$3"
    local comment="$4"

    if id "$user" &>/dev/null; then
	_msg_info "user '$user' is already in the system, reset"
	usermod -g "$group" -G "$supplementary_groups"	\
		-d /var/empty -s "$(command -v nologin)"	\
		-c "$comment" "$user"
    else
	useradd -g "$group" -G "$supplementary_groups"	\
		-d /var/empty -s "$(command -v nologin)"	\
		-c "$comment" --system "$user"
	_msg_pass "user added <$user>"
    fi
}

install_unprivileged_daemon()
{ # Return true when installing guix-daemon running without privileges.
    [ "$INIT_SYS" = systemd ] && \
	grep -q "User=guix-daemon" \
	     ~root/.config/guix/current/lib/systemd/system/guix-daemon.service
}

sys_create_build_user()
{ # Create the group and user accounts for build users.

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    if getent group kvm > /dev/null; then
        _msg_info "group kvm exists and build users will be added to it"
        local KVMGROUP=,kvm
    fi

    if install_unprivileged_daemon
    then
	_msg_info "installing guix-daemon to run as an unprivileged user"

	# Installing guix-daemon to run as a non-root user requires
	# unprivileged user namespaces.
	if [ -f /proc/sys/kernel/unprivileged_userns_clone ] \
	       && [ "$(cat /proc/sys/kernel/unprivileged_userns_clone)" -ne 1 ]
	then
	    echo 1 > /proc/sys/kernel/unprivileged_userns_clone || \
		_err "failed to enable unprivileged user namespaces"

	    _msg_warn "Unprivileged user namespaces were disabled and have been enabled now."
	    _msg_warn "This Linux feature is required by guix-daemon.  To enable it permanently, run:"
	    _msg_warn '  echo 1 > /proc/sys/kernel/unprivileged_userns_clone'
	    _msg_warn "from the relevant startup script."
	fi


	if getent group guix-daemon > /dev/null; then
	    _msg_info "group guix-daemon exists"
	else
	    groupadd --system guix-daemon
	    _msg_pass "group guix-daemon created"
	fi

	create_account guix-daemon guix-daemon		\
		       guix-daemon"$KVMGROUP"		\
		       "Unprivileged Guix Daemon User"

	# ‘tar xf’ creates root:root files.  Change that.
	chown -R guix-daemon:guix-daemon /gnu /var/guix
	chown -R root:root /var/guix/profiles/per-user/root

	# The unprivileged daemon cannot create the log directory by itself.
	mkdir -p /var/log/guix
	chown guix-daemon:guix-daemon /var/log/guix
	chmod 755 /var/log/guix
    else
	if getent group guixbuild > /dev/null; then
            _msg_info "group guixbuild exists"
	else
            groupadd --system guixbuild
            _msg_pass "group <guixbuild> created"
	fi

	for i in $(seq -w 1 10); do
	    create_account "guixbuilder${i}" "guixbuild"	\
	                   "guixbuild${KVMGROUP}"		\
			   "Guix build user $i"
	done
    fi
}

sys_delete_build_user()
{
    for i in $(seq -w 1 10); do
        if id -u "guixbuilder${i}" &>/dev/null; then
            userdel -f guixbuilder"$i"
        fi
    done

    _msg_info "delete group guixbuild"
    if getent group guixbuild &>/dev/null; then
        groupdel -f guixbuild
    fi

    _msg_info "remove guix-daemon user"
    if id guix-daemon &>/dev/null; then
	userdel -f guix-daemon
    fi
    if getent group guix-daemon &>/dev/null; then
	groupdel -f guix-daemon
    fi
}

sys_enable_guix_daemon()
{ # Run the daemon, and set it to automatically start on boot.

    local info_path
    local local_bin
    local var_guix

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    info_path="/usr/local/share/info"
    local_bin="/usr/local/bin"
    var_guix="/var/guix/profiles/per-user/root/current-guix"

    case "$INIT_SYS" in
        upstart)
            { initctl reload-configuration;
              cp ~root/.config/guix/current/lib/upstart/system/guix-daemon.conf \
                 /etc/init/ &&
                  configure_substitute_discovery /etc/init/guix-daemon.conf &&
                  start guix-daemon; } &&
                _msg_pass "enabled Guix daemon via upstart"
            ;;
        systemd)
            { install_unit()
              {
                  local dest="/etc/systemd/system/$1"
                  rm -f "$dest"
                  cp ~root/.config/guix/current/lib/systemd/system/"$1" "$dest"
                  chmod 664 "$dest"
                  systemctl daemon-reload
                  systemctl enable "$1"
              }

              install_unit guix-daemon.service

              configure_substitute_discovery \
                  /etc/systemd/system/guix-daemon.service

              # Install after guix-daemon.service to avoid a harmless warning.
              # systemd .mount units must be named after the target directory.
	      install_unit gnu-store.mount

              systemctl daemon-reload &&
                  systemctl start guix-daemon &&
	          systemctl start gnu-store.mount; } &&
                _msg_pass "enabled Guix daemon via systemd"
            ;;
        sysv-init)
            { mkdir -p /etc/init.d;
              cp ~root/.config/guix/current/etc/init.d/guix-daemon \
                 /etc/init.d/guix-daemon;
              chmod 775 /etc/init.d/guix-daemon;

              configure_substitute_discovery /etc/init.d/guix-daemon

              update-rc.d guix-daemon defaults &&
                  update-rc.d guix-daemon enable &&
                  service guix-daemon start; } &&
                _msg_pass "enabled Guix daemon via sysv"
            ;;
        openrc)
            { mkdir -p /etc/init.d;
              cp ~root/.config/guix/current/etc/openrc/guix-daemon \
                 /etc/init.d/guix-daemon;
              chmod 775 /etc/init.d/guix-daemon;

              configure_substitute_discovery /etc/init.d/guix-daemon

              rc-update add guix-daemon default &&
                  rc-service guix-daemon start; } &&
                _msg_pass "enabled Guix daemon via OpenRC"
            ;;
        NA|*)
            _err "unsupported init system; run the daemon manually:"
            echo "  ~root/.config/guix/current/bin/guix-daemon --build-users-group=guixbuild"
            ;;
    esac

    _msg_info "making the guix command available to other users"

    [ -e "$local_bin" ] || mkdir -p "$local_bin"
    ln -sf "${var_guix}/bin/guix"  "$local_bin"

    [ -e "$info_path" ] || mkdir -p "$info_path"
    for i in "${var_guix}"/share/info/*; do
        ln -sf "$i" "$info_path"
    done
}

sys_delete_guix_daemon()
{ # Disabled, stop and remove the various guix daemons.

    local info_path
    local local_bin
    local var_guix

    _debug "--- [ ${FUNCNAME[0]} ] ---"

    info_path="/usr/local/share/info"
    local_bin="/usr/local/bin"


    case "$INIT_SYS" in
        upstart)
            _msg_info "stopping guix-daemon"
            stop guix-daemon
            _msg_info "removing guix-daemon"
            rm /etc/init/guix-daemon.conf
            ;;

        systemd)
            if [ -f /etc/systemd/system/guix-daemon.service ]; then
                _msg_info "disabling guix-daemon"
                systemctl disable guix-daemon
                _msg_info "stopping guix-daemon"
                systemctl stop guix-daemon
                _msg_info "removing guix-daemon"
                rm -f /etc/systemd/system/guix-daemon.service
            fi

            if [ -f /etc/systemd/system/gnu-store.mount ]; then
                _msg_info "disabling gnu-store.mount"
                systemctl disable gnu-store.mount
                _msg_info "stopping gnu-store.mount"
                systemctl stop gnu-store.mount
                _msg_info "removing gnu-store.mount"
                rm -f /etc/systemd/system/gnu-store.mount
            fi
            systemctl daemon-reload
            ;;

        sysv-init)
            update-rc.d guix-daemon disable
            service guix-daemon stop
            rm -rf /etc/init.d/guix-daemon
            ;;
        NA|*)
            _err "unsupported init system; disable, stop and remove the daemon manually:"
            echo "  ~root/.config/guix/current/bin/guix-daemon --build-users-group=guixbuild"
            ;;
    esac


    _msg_info "removing $local_bin/guix"
    rm -f "$local_bin"/guix

    _msg_info "removing $info_path/guix*"
    rm -f "$info_path"/guix*
}

sys_authorize_build_farms()
{ # authorize the public key(s) of the build farm(s)
    local hosts=(
	bordeaux.guix.gnu.org
	ci.guix.gnu.org
    )

    if prompt_yes_no "Permit downloading pre-built package binaries from the \
project's build farms?"; then
	for host in "${hosts[@]}"; do
	    local key=~root/.config/guix/current/share/guix/$host.pub
	    [ -f "$key" ] \
		&& guix archive --authorize < "$key" \
		&& _msg_pass "Authorized public key for $host"
	done
	if id guix-daemon &>/dev/null; then
	    # /etc/guix/acl must be readable by the unprivileged guix-daemon.
	    chown -R guix-daemon:guix-daemon /etc/guix
	fi
    else
        _msg_info "Skipped authorizing build farm public keys"
    fi
}

sys_create_init_profile()
{ # Define for better desktop integration
  # This will not take effect until the next shell or desktop session!
    [ -d "/etc/profile.d" ] || mkdir /etc/profile.d # Just in case
    cat <<"EOF" > /etc/profile.d/zzz-guix.sh
# Explicitly initialize XDG base directory variables to ease compatibility
# with Guix System: see <https://issues.guix.gnu.org/56050#3>.
export XCURSOR_PATH="${XCURSOR_PATH:-/usr/local/share/icons:/usr/share/icons}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$HOME/.local/state}"
export XDG_DATA_DIRS="${XDG_DATA_DIRS:-/usr/local/share/:/usr/share/}"
export XDG_CONFIG_DIRS="${XDG_CONFIG_DIRS:-/etc/xdg}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
# no default for XDG_RUNTIME_DIR (depends on foreign distro for semantics)

# `guix pull` profile
GUIX_PROFILE="$HOME/.config/guix/current"
export PATH="$GUIX_PROFILE/bin${PATH:+:}$PATH"
# Add to INFOPATH and MANPATH so the latest Guix documentation is available to
# info and man readers.  When INFOPATH is unset, add a trailing colon so Emacs
# searches 'Info-default-directory-list'.  When MANPATH is unset, add a
# trailing colon so the system default search path is used.
export INFOPATH="$GUIX_PROFILE/share/info:$INFOPATH"
export MANPATH="$GUIX_PROFILE/share/man:$MANPATH"
# Expose the latest Guix modules to Guile so guix shell and repls spawned by
# e.g. Geiser work out of the box.
export GUILE_LOAD_PATH="$GUIX_PROFILE/share/guile/site/3.0${GUILE_LOAD_PATH:+:}$GUILE_LOAD_PATH"
export GUILE_LOAD_COMPILED_PATH="$GUIX_PROFILE/lib/guile/3.0/site-ccache${GUILE_LOAD_COMPILED_PATH:+:}$GUILE_LOAD_COMPILED_PATH"

# User's default profile, if it exists
GUIX_PROFILE="$HOME/.guix-profile"
if [ -L "$GUIX_PROFILE" ]; then
  . "$GUIX_PROFILE/etc/profile"

  # see info '(guix) Application Setup'
  export GUIX_LOCPATH="$GUIX_PROFILE/lib/locale${GUIX_LOCPATH:+:}$GUIX_LOCPATH"

  # Documentation search paths may be handled by $GUIX_PROFILE/etc/profile if
  # the user installs info and man readers via Guix.  If the user doesn’t,
  # explicitly add to them so documentation for software from ‘guix install’
  # is available to the system info and man readers.
  case $INFOPATH in
    *$GUIX_PROFILE/share/info*) ;;
    *) export INFOPATH="$GUIX_PROFILE/share/info:$INFOPATH" ;;
  esac
  case $MANPATH in
    *$GUIX_PROFILE/share/man*) ;;
    *) export MANPATH="$GUIX_PROFILE/share/man:$MANPATH"
  esac
fi

# NOTE: Guix Home handles its own profile initialization in ~/.profile. See
# info '(guix) Configuring the Shell'.

# Clean up after ourselves.
unset GUIX_PROFILE
EOF
}

sys_create_shell_completion()
{ # Symlink supported shell completions system-wide

    var_guix=/var/guix/profiles/per-user/root/current-guix
    bash_completion=/etc/bash_completion.d
    zsh_completion=/usr/share/zsh/site-functions
    fish_completion=/usr/share/fish/vendor_completions.d

    { # Just in case
        for dir_shell in $bash_completion $zsh_completion $fish_completion; do
            [ -d "$dir_shell" ] || mkdir -p "$dir_shell"
        done;

        # Don't use globing here as we also need to delete the files when
        # uninstalling Guix
        ln -sf ${var_guix}/etc/bash_completion.d/guix "$bash_completion";
        ln -sf ${var_guix}/etc/bash_completion.d/guix-daemon "$bash_completion";
        ln -sf ${var_guix}/share/zsh/site-functions/_guix "$zsh_completion";
        ln -sf ${var_guix}/share/fish/vendor_completions.d/guix.fish "$fish_completion"; } &&
        _msg_pass "installed shell completion"
}

sys_delete_shell_completion()
{ # Symlink supported shell completions system-wide

    var_guix=/var/guix/profiles/per-user/root/current-guix
    bash_completion=/etc/bash_completion.d
    zsh_completion=/usr/share/zsh/site-functions
    fish_completion=/usr/share/fish/vendor_completions.d

    _msg_info "removing shell completion"

    rm -f "$bash_completion"/guix;
    rm -f "$bash_completion"/guix-daemon;
    rm -f "$zsh_completion"/_guix;
    rm -f "$fish_completion"/guix.fish;
}

sys_customize_bashrc()
{
    prompt_yes_no "Customize users Bash shell prompt for Guix?" || return 0

    for bashrc in /home/*/.bashrc /root/.bashrc; do
        test -f "$bashrc" || continue
        # shellcheck disable=SC2016 # intended search for variable reference
        grep -Fq '$GUIX_ENVIRONMENT' "$bashrc" && continue
        cp "${bashrc}" "${bashrc}.bak"
        # shellcheck disable=SC2016,SC2028 # intended literal shell output
        echo '
# Automatically added by the Guix install script.
if [ -n "$GUIX_ENVIRONMENT" ]; then
    if [[ $PS1 =~ (.*)"\\$" ]]; then
        PS1="${BASH_REMATCH[1]} [env]\\\$ "
    fi
fi
' >> "$bashrc"
    done
    _msg_pass "Bash shell prompt successfully customized for Guix"
}

sys_maybe_setup_selinux()
{
    if ! [ -f /sys/fs/selinux/policy ]
    then
	return
    fi

    local c
    for c in semodule restorecon
    do
        if ! command -v "$c" &>/dev/null
	then
	    return
	fi
    done

    prompt_yes_no "Install SELinux policy that might be required to run guix-daemon?" \
	|| return 0

    local var_guix=/var/guix/profiles/per-user/root/current-guix
    semodule -i "${var_guix}/share/selinux/guix-daemon.cil"
    restorecon -R /gnu /var/guix
}

sys_delete_init_profile()
{
    _msg_info "removing /etc/profile.d/zzz-guix.sh"
    rm -f /etc/profile.d/zzz-guix.sh
}

sys_delete_user_profiles()
{
    _msg_info "removing ~root/.guix-profile"
    rm -f ~root/.guix-profile
    rm -rf ~root/.cache/guix

    _msg_info "removing .guix-profile, .cache/guix and .config/guix of all /home users"
    rm -rf /home/*/{.guix-profile,{.cache,.config}/guix}
}

welcome()
{
    local uninstall_flag="$1"
    local char
    cat<<"EOF"
    ░░░                                     ░░░
    ░░▒▒░░░░░░░░░               ░░░░░░░░░▒▒░░
     ░░▒▒▒▒▒░░░░░░░           ░░░░░░░▒▒▒▒▒░
         ░▒▒▒░░▒▒▒▒▒         ░░░░░░░▒▒░
               ░▒▒▒▒░       ░░░░░░
                ▒▒▒▒▒      ░░░░░░
                 ▒▒▒▒▒     ░░░░░
                 ░▒▒▒▒▒   ░░░░░
                  ▒▒▒▒▒   ░░░░░
                   ▒▒▒▒▒ ░░░░░
                   ░▒▒▒▒▒░░░░░
                    ▒▒▒▒▒▒░░░
                     ▒▒▒▒▒▒░
     _____ _   _ _    _    _____       _
    / ____| \ | | |  | |  / ____|     (_)
   | |  __|  \| | |  | | | |  __ _   _ ___  __
   | | |_ | . ' | |  | | | | |_ | | | | \ \/ /
   | |__| | |\  | |__| | | |__| | |_| | |>  <
    \_____|_| \_|\____/   \_____|\__,_|_/_/\_\

https://www.gnu.org/software/guix/
EOF

    if [ '--uninstall' = "$uninstall_flag" ]; then
        echo "${WARN}This script will uninstall GNU Guix from your system"
        echo "To install, run this script with no parameters."
    else
        echo "This script installs GNU Guix on your system"
        echo "To uninstall, pass in the '--uninstall' parameter."
    fi

    # Don't use ‘read -p’ here!  It won't display when run non-interactively.
    echo -n "Press return to continue..."$'\r'
    if ! read -r char; then
	echo
	die "Can't read standard input.  Hint: don't pipe scripts into a shell."
    fi
    if [ "$char" ]; then
	echo
	echo "...that ($char) was not a return!"
	_msg_warn "Use newlines to automate installation, e.g.: yes '' | ${0##*/}"
	_msg_warn "Any other method is unsupported and likely to break in future."
    fi
}

main_install()
{
    local tmp_path
    welcome

    _msg "Starting installation ($(date))"

    chk_term
    chk_init_sys
    add_init_sys_require
    chk_require "${REQUIRE[@]}"
    chk_gpg_keyring
    chk_sys_arch
    chk_sys_nscd
    chk_existing

    _msg_info "system is ${ARCH_OS}"

    umask 0022
    tmp_path="$(mktemp -t -d guix.XXXXXX)"

    if [ -z "${GUIX_BINARY_FILE_NAME}" ]; then
        guix_get_bin_list "${GNU_URL}"
        guix_get_bin "${GNU_URL}" "${BIN_VER}" "$tmp_path"
        GUIX_BINARY_FILE_NAME=${BIN_VER}.tar.xz
    else
        if ! [[ $GUIX_BINARY_FILE_NAME =~ $ARCH_OS ]]; then
            _err "$ARCH_OS not in ${GUIX_BINARY_FILE_NAME}; aborting"
        fi
        _msg_info "Using manually provided binary ${GUIX_BINARY_FILE_NAME}"
        GUIX_BINARY_FILE_NAME=$(realpath "$GUIX_BINARY_FILE_NAME")
    fi

    sys_create_store "${GUIX_BINARY_FILE_NAME}" "${tmp_path}"
    sys_create_build_user
    sys_maybe_setup_selinux
    sys_enable_guix_daemon
    sys_authorize_build_farms
    sys_create_init_profile
    sys_create_shell_completion
    sys_customize_bashrc

    _msg_info "cleaning up ${tmp_path}"
    rm -r "${tmp_path}"

    _msg_pass "Guix has successfully been installed!"
    _msg_info "Run 'info guix' to read the manual."

    # Required to source /etc/profile in desktop environments.
    _msg_info "Please log out and back in to complete the installation."
 }

main_uninstall()
{
    welcome --uninstall
    _msg "Starting uninstall process ($(date))"

    chk_term
    chk_require "${REQUIRE[@]}"
    # it's ok to leave the gpg key
    chk_init_sys
    chk_sys_arch

    _msg_info "system is ${ARCH_OS}"

    # stop the build, package system.
    sys_delete_guix_daemon
    # stop people from accessing their profiles.
    sys_delete_user_profiles
    # kill guix off all the guts of guix
    sys_delete_store
    # clean up the system
    sys_delete_init_profile
    sys_delete_build_user
    sys_delete_shell_completion

    # these directories are created on the fly during usage.
    _msg_info "removing /etc/guix"
    rm -rf /etc/guix
    _msg_info "removing /var/log/guix"
    rm -rf /var/log/guix

    _msg_pass "Guix has successfully been uninstalled!"
}

main()
{
    # expect no parameters
    # or '--uninstall'
    if [ 0 -eq $# ]; then
        main_install
    else
        local uninstall_flag="$1"
        if [ '--uninstall' = "${uninstall_flag}" ]; then
            main_uninstall
        else
            echo "unsupported parameters: $*"
            exit 1
        fi
    fi
}

main "$@"
