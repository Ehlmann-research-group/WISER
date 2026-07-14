# Code-signing WISER on Windows

Code-signing is a widely-accepted technique for verifying that distributable
binaries come from a known source and have not been tampered with since they
were built. On Windows, an unsigned installer triggers a Microsoft Defender
SmartScreen warning ("Windows protected your PC"), and many browsers and
anti-virus tools will flag the download. Code-signing the WISER installer and
uninstaller removes these warnings and lets users trust that the application
came from us.

This document describes how WISER is code-signed on Windows, what tools you
need installed, and how the installer script itself works.

Unlike macOS, Windows code-signing does not require a separate notarization
step. You build the installer and sign it in a single `make` command, and the
signature is embedded directly into the `.exe`.

## Tools You Need

To build and code-sign the WISER installer on Windows you need two pieces of
software installed locally — **NSIS** and the **Windows SDK** — plus the
physical code-signing key.

### NSIS

The [Nullsoft Scriptable Install System (NSIS)](https://nsis.sourceforge.io/Main_Page)
compiles `install-win/win-install.nsi` into the WISER installer. Download the
installer from the [NSIS download page](https://nsis.sourceforge.io/Download)
and run it; the default options are fine. After installation, `makensis.exe` is
typically located at `C:\Program Files (x86)\NSIS\makensis.exe`, which is the
path the WISER `Makefile` expects (see the `NSIS` variable near the top of the
`Makefile`).

### Windows SDK (signtool)

The actual signing is performed by `signtool.exe`, which ships with the
[Windows 10/11 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/).
Download and run the SDK installer. You only need the **"Windows SDK Signing
Tools for Desktop Apps"** component, though installing the full SDK is fine.

After installation, `signtool.exe` lives under a versioned path such as:

```
C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe
```

The version number in that path (`10.0.26100.0` above) depends on which SDK
version you installed, so it will likely differ on your machine. See
[Configuring the Signtool Path](#configuring-signtool) below for how to find
your exact path and point the install script at it.

### Code-signing certificate (USB token)

WISER is signed with a certificate stored on a physical USB hardware token
(for example a YubiKey or a SafeNet/eToken device). To sign, you must:

1. Plug the USB token into your machine.
2. Install the token vendor's driver/middleware (e.g. SafeNet Authentication
   Client) so Windows can talk to it. Once installed, the token's certificate
   is automatically registered in the Windows Certificate Store.
3. Know the certificate's **SHA-1 thumbprint** (see
   [Finding Your Certificate Thumbprint](#finding-thumbprint)), which is how
   the install script selects the right certificate.

Because the script identifies the certificate by thumbprint and resolves it
from the Windows Certificate Store at signing time, the only machine-specific
secret you need is the thumbprint — there is nothing else to copy off of the
original build machine.

## Building and Signing: the `make dist-win` command

Currently, we let the `Build and Smoke Test WISER` action build the Windows
installer, and we sign it locally by running
`make sign-windows LINK=<link-to-windows-artifact>`. We get the link to the
Windows artifact by right-clicking on `wiser-Windows-X64` and selecting
_Copy Link Address_. Once the installer is signed, we upload it.

You can also build and sign in a single Make target. Because building is so
machine-dependent, however, we prefer to build on the runners using the steps
above. If you still want to build and sign locally, activate the `wiser-prod`
conda environment, change to the root directory of WISER, and run:

```
make dist-win
```

This first runs the `build-win` target (which freezes WISER with PyInstaller
into `dist/WISER`) and then invokes NSIS. The build-and-sign line in the
`Makefile` is:

```
$(NSIS) /NOCD /DWISER_VERSION="$(APP_VERSION)" /DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT) install-win\win-install.nsi
```

Here is what each part does:

*   **`$(NSIS)`** — the path to the NSIS compiler, `makensis.exe`. It is
    defined once near the top of the `Makefile` as
    `NSIS="C:\Program Files (x86)\NSIS\makensis.exe"`. This is the program that
    reads the `.nsi` script and produces the installer `.exe`.

*   **`/NOCD`** — "no change directory." By default `makensis` changes its
    working directory to the folder containing the script. `/NOCD` tells it to
    stay in the directory where `make` was invoked (the repository root) so
    that the relative paths inside the script — like `dist\WISER\*.*` and
    `install-win\license.rtf` — resolve correctly.

*   **`/DWISER_VERSION="$(APP_VERSION)"`** — defines the `WISER_VERSION` symbol
    inside the script. The `/D` flag is NSIS's "define" option, analogous to a
    compiler's `-D`. `$(APP_VERSION)` is computed by the `Makefile` by running
    `python src/wiser/version.py`, so the version is never hard-coded. The
    script uses `WISER_VERSION` to name the install directory, the registry
    keys, and the output file (e.g. `Install-WISER-1.4b1.exe`). The script
    deliberately `!error`s out if this symbol is not provided.

*   **`/DSHA1_THUMBPRINT=$(SHA1_THUMBPRINT)`** — defines the `SHA1_THUMBPRINT`
    symbol: the SHA-1 thumbprint of your code-signing certificate.
    `$(SHA1_THUMBPRINT)` comes from your `Secret.mk` file — a copy of
    `Secret-template.mk` that is **not** checked into git. `signtool` uses this
    thumbprint to pick the correct certificate out of the Windows Certificate
    Store. The script also `!error`s out if this symbol is not provided.

*   **`install-win\win-install.nsi`** — the NSIS script to compile. This is the
    last argument, and is the script described in
    [How the Install Script Works](#how-the-install-script-works) below.

The related `quick-sign-win` target runs the exact same `makensis` command
*without* first rebuilding with PyInstaller. It is handy when `dist/WISER` is
already built and you only want to (re)build and sign the installer.

(configuring-signtool)=
## Configuring the Signtool Path

The `win-install.nsi` script needs to know where `signtool.exe` lives on your
machine. This is set near the top of the script with a single define:

```
!define SIGN_TOOL "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool"
```

Because the SDK version number in that path differs between machines, you will
likely need to change it. To find your own `signtool.exe`:

1. Open File Explorer and navigate to
   `C:\Program Files (x86)\Windows Kits\10\bin`.
2. You will see one or more version folders (e.g. `10.0.22621.0`,
   `10.0.26100.0`). Open the newest one.
3. Open the `x64` subfolder. `signtool.exe` should be inside.
4. Copy that folder path (without the trailing `.exe`) into the `SIGN_TOOL`
   define.

Alternatively, from a PowerShell prompt you can search for it directly:

```
Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter signtool.exe |
  Where-Object { $_.FullName -like "*x64*" }
```

Use the path to the `x64` build for 64-bit Windows.

(finding-thumbprint)=
## Finding Your Certificate Thumbprint

The `SHA1_THUMBPRINT` value comes from your code-signing certificate. With the
USB token plugged in and its middleware installed, list the certificates in
your personal store from PowerShell:

```
Get-ChildItem Cert:\CurrentUser\My | Format-List Subject, Thumbprint
```

Find the WISER code-signing certificate in the output and copy its
`Thumbprint` value (a 40-character hex string). Put it in your `Secret.mk`:

```
SHA1_THUMBPRINT=your.sha1.thumbprint.here
```

The thumbprint is a property of the certificate itself, so it is the same on
any machine the token is plugged into. It is the only signing secret you need
to carry between machines.

(how-the-install-script-works)=
## How the Install Script Works

`install-win/win-install.nsi` does two jobs that have to be carefully ordered
so that **both** the installer and the uninstaller end up code-signed. NSIS
normally embeds the uninstaller into the installer at compile time, but an
uninstaller created that way cannot itself be signed. To work around this, the
script uses the
[two-pass technique from the NSIS wiki](https://nsis.sourceforge.io/Signing_an_Uninstaller):

1.  **Inner pass.** The script invokes `makensis` on itself a second time with
    the `INNER` symbol defined (see the `!makensis` call in the outer block).
    When `INNER` is defined, the script's only job is to write the uninstaller
    binary out to `%TEMP%` and quit. The outer pass then runs that temporary
    installer, which drops a standalone `Uninstall WISER-<ver>.exe`.

2.  **Sign the uninstaller.** Back in the outer pass, the script calls
    `${SIGN_TOOL}` to code-sign the just-created uninstaller:

    ```
    "${SIGN_TOOL}" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%TEMP%\Uninstall WISER-<ver>.exe"
    ```

3.  **Build the real installer.** The outer pass continues, packaging the
    frozen `dist\WISER` tree, the icon, and the now-signed uninstaller into the
    final `Install-WISER-<ver>.exe`.

4.  **Sign the installer.** The `!finalize` directive runs after the installer
    is written and signs it the same way (`%1` is the installer filename that
    NSIS just produced).

The signing commands themselves are straightforward:

*   **`sign /sha1 "${SHA1_THUMBPRINT}"`** selects the certificate by thumbprint
    from the Windows Certificate Store (populated by the USB token's
    middleware).
*   **`/fd SHA256`** sets the file digest algorithm to SHA-256.
*   **`/t http://timestamp.sectigo.com`** adds a trusted timestamp so the
    signature stays valid even after the signing certificate eventually
    expires.

Both signing commands reference the single `SIGN_TOOL` define, so updating that
one line (see [Configuring the Signtool Path](#configuring-signtool)) is enough
to point the whole script at your `signtool.exe`.
