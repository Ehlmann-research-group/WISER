# Code-signing WISER on the Mac

Code-signing has become a widely-accepted technique for verifying that
distributable binaries are from a known source, and also that they have
not been tampered with since they were built.  On the macOS platform,
code-signing is of even greater importance, since the Gatekeeper subsystem
will not allow a downloaded application binary to be run _at all_, unless
the binary was code-signed with a known developer key, and also notarized
by Apple.  This is a daunting hurdle to overcome, but it is surmountable.

This document describes how to code-sign and notarize an application for
MacOSX, using the WISER application as an example.

## The full sequence

Three separate operations are needed, in this order, and all three are required before a
downloaded WISER opens without a Gatekeeper prompt:

1.  **Code-sign** the `.app` with the Developer ID Application certificate (`codesign`).
2.  **Notarize** it — upload to Apple's malware scan, which returns a ticket
    (`xcrun notarytool`).
3.  **Staple** the ticket to the artifact so Gatekeeper can validate it *offline*
    (`xcrun stapler`).

The `.app` must be stapled **before** the DMG is built, because an app sitting on a
read-only disk image cannot be stapled afterwards. The DMG is then signed, notarized and
stapled in turn. `src/devtools/sign_mac.py` performs the whole sequence:

    sign .app -> notarize -> staple .app -> build DMG -> sign DMG -> notarize -> staple DMG -> verify

The rest of this document explains each operation. To actually run it, use the GitHub
Action (below) or `make sign-mac`; the sections after that cover what those do.

## Code-signing by Identified Developers

In order to code-sign an application, one must have an Identified Developer
key from Apple.  This requires signing up for the Apple Developer program,
a $99/yr expense.  Open the [Apple Developer](https://developer.apple.com/)
site in a web browser, log in with your Apple ID, and pay to join the
developer program.

Once your developer account is active, you should be able to navigate to the
[Certificates, Identifiers & Profiles](https://developer.apple.com/account/resources/certificates/list)
page to see your certificates.

1.  Click the blue "+" symbol to the right of the "Certificates" header
    to create a new certificate.  This will take you to a new page with
    the header "Create a New Certificate."

2.  The "Create a New Certificate" page presents a list of different
    certificate types that you can request.  To code-sign applications
    as an Identified Developer, select the "Developer ID Application"
    option at the bottom of the list.

    Then, select the blue "Continue" button back at the top of the page,
    to go to the next step.

3.  The next page in the "Create a New Certificate" workflow asks for a
    Certificate Signing Request (CSR).  You can follow the instructions
    linked from this page.

    You can specify an Email Address and Common Name associated with your
    project, or you can specify your own email and name for this key.
    This information doesn't seem to be reported to the user anywhere, so
    in one sense it's somewhat immaterial.

    The main thing is, the key's name will likely be your name from your
    Apple ID account.

    Once you have uploaded the CSR to this page,

4.  Once you have created and uploaded a CSR, you can advance to the next
    page, which will allow you to download your code-signing key.

    When you download the key, your browser will likely ask if you want to
    open the key in the Keychain Access tool that comes with macOS (in the
    Applications/Utilities folder).  Go ahead and import it.

## Preparing to Notarize from the Command Line

To notarize a Mac application, it must be uploaded to Apple servers using
your Apple ID login credentials.  Apple suggests that you create an
_app-specific password_ for the notarization step, so that you don't have
to use your main account password.

You can go to [this Apple Support page](https://support.apple.com/en-us/HT204397)
to learn how to create an app-specific password.  **Note that the password
will only be shown to you once, so don't forget it!**

>   I recommend storing this app-specific password in a "secret" config
>   file (i.e. the file contains secrets, not that the file is hidden),
>   that is _NOT_ checked in to your code repository.  This way you can
>   automate your build, code-sign and notarization process without
>   sharing your app-specific password or other information with others.

## Code-Signing Your Application

After your program has been built into a distributable form (e.g. by freezing
a Python application, or building your application's binaries), but _before_
it has been packaged into a Disk Image (.dmg) file, it must be code-signed
using the Identified Developer key generated earlier.

This is an example of how you could codesign:

```
codesign -s $(AD_CODESIGN_KEY_NAME) --deep --force \
        --entitlements install-mac/entitlements.plist \
        -o runtime dist/$(APP_NAME).app
```

However, the use of --deep is highly discouraged and has led to problems in
WISER development so we instead use a script to recursively search through
the WISER.app folder to sign files.

In the WISER Makefile, we simply run this script
```
bash install-mac/sign_wiser.sh
```

Here are descriptions of the relevant arguments `codesign`:

*   The `AD_CODESIGN_KEY_NAME` is the name of the Identified Developer
    key.  This is typically the name of the Apple ID account owner.

*   The `APP_NAME` is simply the WISER application name (defined to be
    "WISER" in this `Makefile`).

*   The `--deep` argument is often necessary, particularly when the
    application includes shared libraries (`.dylib` files) or other
    binary programs.  If these additional binaries are not code-signed
    (e.g. by their original provider, or by your build process), then
    Apple's notarization mechanism will complain.  The easy way to deal
    with this situation is to just ensure that everything is code-signed
    using your Identified Developer code-signing key.

*   The `--force` argument is also necessary when shared libraries _are_
    signed by their original providers.  In these cases, the `codesign` tool
    will not replace the existing signature unless the command-line also
    specifies this argument - and this will cause the notarization step to
    fail.

*   The `-o runtime` argument specifies the use of the
    [hardened runtime](https://developer.apple.com/documentation/security/hardened_runtime),
    which prevents certain kinds of exploits.  **This option must be
    specified for Apple notarization to succeed.**

*   The `--entitlements` argument specifies various entitlements (i.e.
    extra permissions) that should be granted to the application.  For
    frozen Python applications, certain entitlements must be specified,
    or else the application will not successfully run after installation.

    The version from the `install-mac` directory is as follows:

    ```
    <!-- From https://github.com/pyinstaller/pyinstaller/issues/4629 -->
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
    	<!-- These are required for binaries built by PyInstaller -->

        <!-- supposedly unneeded
    	<key>com.apple.security.cs.allow-jit</key>
    	<true/>
        -->

    	<key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    	<true/>
    </dict>
    </plist>
    ```

    Searching on the Internet can help resolve issues related to entitlements.
    In particular, see the bug-report linked from the `entitlements.plist`
    file above.


## Signing in GitHub Actions

Releases are signed by the **Sign macOS release assets** workflow
(`.github/workflows/sign-macos.yml`) rather than from a laptop.  Dispatch it with the run
ID of a completed "Build and Smoke Test WISER" run, and optionally a release tag:

- `run_id` — the build run whose macOS artifact should be signed.
- `arch` — `both` (default), `arm64`, or `x64`.
- `release_tag` — when set, the signed DMG is attached to that release.  Leave it empty to
  produce a workflow artifact only.

The workflow refuses to sign a run that did not conclude successfully, and — when a tag is
given — refuses when the tag does not point at the commit that was built.  It checks the
repository out at the build commit, so `version.py` names the DMG for the artifact being
signed rather than for whatever `main` has since moved on to.

It runs the same `src/devtools/sign_mac.py` as `make sign-mac`, so the local and CI paths
cannot drift.

### Required configuration

The signing job runs in a **`release` environment**.  Configure that environment with a
required reviewer: the certificate is unreachable until a maintainer approves the
deployment, which is what keeps a private key in GitHub acceptable.

Environment secrets:

| Secret | Purpose |
| --- | --- |
| `MACOS_CERT_P12_BASE64` | Developer ID Application certificate + key, as base64 of a `.p12` |
| `MACOS_CERT_PASSWORD` | Password protecting that `.p12` |
| `NOTARY_KEY_P8_BASE64` | App Store Connect API key, as base64 of the `.p8` (preferred) |
| `NOTARY_KEY_ID` | Key ID for that API key |
| `NOTARY_ISSUER_ID` | Issuer ID; omit for an individual (non-team) key |
| `AD_USERNAME`, `AD_TEAM_ID`, `AD_PASSWORD` | Apple ID fallback, used when no API key is set |

Export the `.p12` from Keychain Access (right-click the certificate, Export), then:

```
base64 -i Certificates.p12 | pbcopy
```

The signing identity is **not** configured anywhere: the job reads it back out of the
keychain it imports, and fails if that keychain does not contain exactly one code-signing
identity.  A re-issued certificate therefore needs no workflow change.

An App Store Connect API key is preferred over the Apple ID credentials because it is
scoped to the team rather than to one person's login and 2FA, and can be revoked on its
own.  The workflow uses the API key when `NOTARY_KEY_P8_BASE64` is set and falls back to
the Apple ID otherwise.

It also keeps the credential out of process arguments.  `notarytool` accepts an
app-specific password only as `--password`, or through a keychain profile that
`store-credentials` populates the same way, so on the Apple ID path the password is
visible in `ps` for the duration of the submission.  An API key is passed as a *file
path*, and the Key ID and Issuer ID are identifiers rather than secrets, so nothing
sensitive appears in the command line.

## Building a Disk Image (.dmg) File

It is of value to build a distributable `.dmg` file from the application,
so that it's simple to share with others.  The WISER build process uses
these steps:

```
hdiutil create dist/tmp.dmg -ov -volname "$(APP_NAME)" -fs HFS+ -srcfolder dist/$(APP_NAME).app
hdiutil convert dist/tmp.dmg -format UDZO -o dist/$(APP_NAME)-$(APP_VERSION).dmg
rm dist/tmp.dmg
```

The first step generates a temporary disk image `tmp.dmg` that mounts with the
specified application name, and with the specified contents.

Note that the first step creates a read-write disk image.  Thus, the second
step is to take the `tmp.dmg` temporary image and prepare a read-only image
suitable for distribution.  Thus, this step writes its output to a filename
suitable for distribution.  The WISER build process generates a filename of
`WISER-<versioninfo>.dmg` (e.g. `WISER-1.0a4-dev0.dmg`), mounted with the
volume name `WISER`.

## Notarizing the Disk Image

This is possibly the most exciting and mysterious step of the entire process.
Once the Disk Image is generated from the previous step, it must be uploaded
to Apple for notarization. This operation requires a command like this:

```
xcrun notarytool submit dist/$(APP_NAME)-$(APP_VERSION).dmg \
    --apple-id $(AD_USERNAME) \
    --team-id $(AD_TEAM_ID) \
    --password $(AD_PASSWORD)
```

Here are descriptions of the relevant arguments:

*   The `--submit` argument specifies the distributable disk-image file that
    you are requesting notarization for.

    **Note that the disk image file will be uploaded to Apple for this
    notarization process to be completed.**  After that, it must be scanned
    by Apple's servers.  Thus, the command will take some time to finish the
    upload, and a successful upload does _NOT_ mean that the app has been
    successfully notarized.

    Note that the bundle-id is automatically retrieved from the disk image
    we upload so we don't need to specify it.

*   The `AD_USERNAME` is the Apple ID of the account owner.  In the WISER
    build process, this configuration is specified in the secret config file.

*   The `AD_TEAM_ID` is the 10-character Developer Team ID. It is required      
    when authenticating with Apple ID credentials.

*   The `AD_PASSWORD` is the app-specific password you generated in an earlier
    step.  In the WISER build process, this is also specified in the secret
    config file.

Previously, `xcrun atool --notarize` was used, but Apple deprecated `atool`
for notarization.

### Output of Notarization

When the notarization command completes, it will output some information that
can be used to monitor the notarization process.  Here is some example output:

```
xcrun notarytool submit dist/WISER-1.3b1.dmg \
                --apple-id <secret> \
                --team-id <secret>
                --password <secret>

Conducting pre-submission checks for WISER-1.3b1.dmg and initiating connection to the Apple notary service...
Submission ID received
  id: 21602dbf-e391-4609-96e2-efc4fec25633
Upload progress: 100.00% (250 MB of 250 MB)    
Successfully uploaded file
  id: 21602dbf-e391-4609-96e2-efc4fec25633
  path: /Users/joshuagk/Documents/WISER/dist/WISER-1.3b1.dmg
```

The most important detail is the _id_ of the request, which can be used to
monitor the status of the notarization process.  The `notarytool` program has a
second command `info` which can be used to fetch the status
of the notarization operation from the Apple servers.

You can run a command like this to fetch the notarization status (substitute
in the `RequestUUID` value above, and specify your Apple ID, team-id, and app-specific password):

```
xcrun notarytool info <request-id> --apple-id <secret> --team-id <secret> --password <secret>
```

I like to run this in a loop so I don't have to manually check on the
process:

```
while true ; do clear ; xcrun notarytool info <request-id> --apple-id <secret> --team-id <secret> --password <secret> ; sleep 10 ; done
```

If everything is correct, you will eventually see something like this:

```
Successfully received submission info
  createdDate: 2025-09-30T20:01:36.033Z
  id: 21602dbf-e391-4609-96e2-efc4fec25633
  name: WISER-1.3b1.dmg
  status: Accepted
```

Regardless of whether it succeeds or fails, once the notarization check is
finished, you should be able to navigate to the run the below command to
get detailed information about what happened during notarization.

```
xcrun notarytool log <request-id>
 --apple-id=<secret> --team-id=<secret> --password=<secret>
```

## Stapling the Ticket

Notarization records the result on Apple's servers; it does not change the file you
uploaded.  Left there, Gatekeeper has to contact Apple over the network the first time a
user launches the app.  That fails for a user who is offline or behind a captive portal,
which is a realistic situation for the classroom and field machines WISER runs on.

`stapler` attaches the notarization ticket to the artifact itself, so Gatekeeper can
validate it with no network at all:

```
xcrun stapler staple dist/WISER.app
xcrun stapler validate dist/WISER.app
```

Staple the `.app` **before** building the DMG.  Once the app is inside a read-only disk
image it can no longer be modified, so an app stapled only at the DMG level ships without
a ticket of its own and phones home on first launch.  Staple the DMG too, after its own
notarization, so the disk image validates offline as well.

## Verifying the Result

```
codesign --verify --strict --deep --verbose=2 dist/WISER.app
xcrun stapler validate dist/WISER.app
xcrun stapler validate dist/WISER-<version>-macos-<arch>.dmg
spctl --assess --type exec --verbose=4 dist/WISER.app
```

The `spctl` assessment is the one that reflects what an end user actually sees; it should
report `source=Notarized Developer ID`.  **It is only meaningful when Gatekeeper
assessments are enabled** — on a machine where `spctl --status` says `assessments
disabled`, `spctl` reports "accepted" for anything, including an unsigned binary.  Check
`spctl --status` first, and re-enable with `sudo spctl --master-enable` when testing.
`sign_mac.py` makes this an error rather than a false pass when run with
`--require-gatekeeper` (as CI does).

### Testing

If you want to test your code-signed and notarized application, you will
need to upload it to some remote server, then download it to your computer
and try to run it (or install it into your Applications folder and then
run it).  Non-code-signed applications can be built and run locally; as
long as they weren't downloaded from somewhere, macOS will run them.  So,
to perform a real test of your code-signed and notarized application, you
will need to put it on some remote system, then download it and attempt to
use it.
