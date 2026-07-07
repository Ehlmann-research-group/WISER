;============================================================================
; To code-sign the uninstaller, this file makes use of this technique:
; https://nsis.sourceforge.io/Signing_an_Uninstaller

; !verbose 4

;--------------------------------
; Headers

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

;--------------------------------
; Project Settings

Name "Workbench for Imaging Spectroscopy Exploration and Research"
Unicode True
ManifestDPIAware True

!define MUI_ICON "icons\wiser.ico"

!ifndef WISER_VERSION
  !error "ERROR:  WISER_VERSION must be defined on NSIS command-line with /D option"
!endif

!ifndef SHA1_THUMBPRINT
  !error "ERROR:  SHA1_THUMBPRINT must be defined on NSIS command-line with /D option"
!endif

; Path to signtool.exe (from the Windows SDK) used to code-sign the installer
; and uninstaller. The Windows SDK version number embedded in this path
; (e.g. 10.0.26100.0) will differ between machines, so you will likely need to
; change this to match the signtool location on your own machine.
;
; For help finding your signtool path, see the "Configuring the Signtool Path"
; section of the Windows code-signing guide: doc/codesign-win.md
!define SIGN_TOOL "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool"

!define APP_BASENAME "WISER"
!define APP_DIRNAME "${APP_BASENAME}-${WISER_VERSION}"
!define APP_DIRNAME_LOWER "wiser-${WISER_VERSION}"
!define UNINSTALL_EXE_NAME "Uninstall ${APP_DIRNAME}.exe"
!define STARTMENU_DIRNAME "${APP_DIRNAME}"
!define STARTMENU_APP_LINK "${APP_DIRNAME}.lnk"
!define STARTMENU_UNINSTALL_LINK "Uninstall ${APP_DIRNAME}.lnk"

; --- Per-user defaults (no admin) ---
RequestExecutionLevel user

InstallDir "$LOCALAPPDATA"

!define REGKEY_UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_DIRNAME}"

Var InstallScope
Var RelaunchArgs
Var SkipDirectoryPage


!ifdef INNER
  !echo "--- Inner Invocation ---"          ; just to see what's going on
  OutFile "$%TEMP%\tempinstaller.exe"       ; not really important where this is
  SetCompress off                           ; for speed
!else
  !echo "--- Outer Invocation ---"

  ; Call makensis again against current file, defining INNER.  This writes an installer for us which, when
  ; it is invoked, will just write the uninstaller to some location, and then exit.

  !echo "Performing inner invocation..."
  !makensis '/NOCD /DINNER /DWISER_VERSION="${WISER_VERSION}" /DSHA1_THUMBPRINT="${SHA1_THUMBPRINT}" "install-win\win-install.nsi"' = 0

  ; So now run that installer we just created as %TEMP%\tempinstaller.exe.  Since it
  ; calls quit the return value isn't zero.

  !system 'set __COMPAT_LAYER=RunAsInvoker&"$%TEMP%\tempinstaller.exe"' = 2

  ; That will have written an uninstaller binary for us.  Now we sign it with your
  ; favorite code signing tool.

  !system '"${SIGN_TOOL}" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%TEMP%\${UNINSTALL_EXE_NAME}"' = 0

  ; Good.  Now we can carry on writing the real installer.

  OutFile "WISER-${WISER_VERSION}-windows-x64-setup.exe"
  ; SetCompressor /SOLID lzma
!endif


 Function .onInit
!ifdef INNER

  ; If INNER is defined, then we aren't supposed to do anything except write out
  ; the uninstaller.  This is better than processing a command line option as it means
  ; this entire code path is not present in the final (real) installer.
  SetSilent silent
  WriteUninstaller "$%TEMP%\${UNINSTALL_EXE_NAME}"
  Quit  ; just bail out quickly when running the "inner" installer
!endif

  StrCpy $InstallScope "current"
  StrCpy $SkipDirectoryPage 0
  SetShellVarContext current

  ${GetParameters} $0
  ${GetOptions} $0 "/ALLUSERS=" $1
  ${If} $1 == "1"
    StrCpy $InstallScope "all"
    StrCpy $SkipDirectoryPage 1
    SetShellVarContext all
  ${EndIf}

  ${GetOptions} $0 "/INSTALLDIR=" $2
  ${If} $2 != ""
    StrCpy $INSTDIR $2
  ${EndIf}
FunctionEnd

!ifndef INNER
!finalize '"${SIGN_TOOL}" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%1"' = 0
!endif

;--------------------------------
; Modern UI 2 Specification

; Installer

!define MUI_LICENSEPAGE_CHECKBOX
!ifndef INNER
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipLicensePageIfElevated
!insertmacro MUI_PAGE_LICENSE "install-win\license.rtf"
  !ifdef MUI_PAGE_CUSTOMFUNCTION_PRE
    !undef MUI_PAGE_CUSTOMFUNCTION_PRE
  !endif
!else
!insertmacro MUI_PAGE_LICENSE "install-win\license.rtf"
!endif
!ifndef INNER
!define MUI_PAGE_CUSTOMFUNCTION_PRE SkipDirectoryPageIfElevated
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE ValidateInstallDir
!insertmacro MUI_PAGE_DIRECTORY
  !ifdef MUI_PAGE_CUSTOMFUNCTION_PRE
    !undef MUI_PAGE_CUSTOMFUNCTION_PRE
  !endif
  !ifdef MUI_PAGE_CUSTOMFUNCTION_LEAVE
    !undef MUI_PAGE_CUSTOMFUNCTION_LEAVE
  !endif
!endif
!insertmacro MUI_PAGE_INSTFILES

; Uninstaller

!ifdef INNER
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!endif

; Language

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Section

Function IsBlockedInstallPath
  StrCpy $0 0

  ${If} "$INSTDIR" == "$WINDIR"
    StrCpy $0 1
    Return
  ${EndIf}
  StrLen $2 "$WINDIR\"
  StrCpy $3 "$INSTDIR" $2
  ${If} "$3" == "$WINDIR\"
    StrCpy $0 1
    Return
  ${EndIf}

  ${If} "$INSTDIR" == "$SYSDIR"
    StrCpy $0 1
    Return
  ${EndIf}
  StrLen $2 "$SYSDIR\"
  StrCpy $3 "$INSTDIR" $2
  ${If} "$3" == "$SYSDIR\"
    StrCpy $0 1
    Return
  ${EndIf}
FunctionEnd

Function NeedsElevation
  StrCpy $0 0

  ${If} "$INSTDIR" == "$PROGRAMFILES"
    StrCpy $0 1
    Return
  ${EndIf}
  StrLen $1 "$PROGRAMFILES\"
  StrCpy $2 "$INSTDIR" $1
  ${If} "$2" == "$PROGRAMFILES\"
    StrCpy $0 1
    Return
  ${EndIf}

  ${If} "$INSTDIR" == "$PROGRAMFILES64"
    StrCpy $0 1
    Return
  ${EndIf}
  StrLen $1 "$PROGRAMFILES64\"
  StrCpy $2 "$INSTDIR" $1
  ${If} "$2" == "$PROGRAMFILES64\"
    StrCpy $0 1
    Return
  ${EndIf}
FunctionEnd

Function NormalizeInstallDir
  ${GetFileName} "$INSTDIR" $0
  ${If} "$0" != "${APP_DIRNAME}"
    ${AndIf} "$0" != "${APP_DIRNAME_LOWER}"
    StrCpy $INSTDIR "$INSTDIR\${APP_DIRNAME}"
  ${EndIf}
FunctionEnd

Function LaunchElevatedAllUsers
  StrCpy $RelaunchArgs '/ALLUSERS=1 /INSTALLDIR="$INSTDIR"'
  ClearErrors
  ExecShell "runas" "$EXEPATH" $RelaunchArgs SW_SHOWNORMAL $0
  ${If} ${Errors}
    MessageBox MB_ICONSTOP|MB_TOPMOST "Unable to relaunch installer as administrator. Exec Shell Code: $0"
    Quit
  ${EndIf}
  Quit
FunctionEnd

Function SkipLicensePageIfElevated
  ${If} $SkipDirectoryPage == 1
    Abort
  ${EndIf}
FunctionEnd

Function SkipDirectoryPageIfElevated
  ${If} $SkipDirectoryPage == 1
    Abort
  ${EndIf}
FunctionEnd

Function ValidateInstallDir
  StrCpy $2 $INSTDIR
  ${GetRoot} "$2" $3
  ${If} "$2" == "$3"
    MessageBox MB_ICONSTOP|MB_TOPMOST "Please choose a folder, not a drive root."
    Abort
  ${EndIf}

  Call NormalizeInstallDir

  Call IsBlockedInstallPath
  ${If} $0 == 1
    MessageBox MB_ICONSTOP|MB_TOPMOST "Please choose a different folder. Installing into drive roots, Windows, or System32 paths is not allowed."
    Abort
  ${EndIf}

  Call NeedsElevation
  ${If} $0 == 1
    Call LaunchElevatedAllUsers
  ${EndIf}

  ClearErrors
  CreateDirectory "$INSTDIR"
  ${If} ${Errors}
    MessageBox MB_ICONSTOP|MB_TOPMOST "Unable to create install directory:$\r$\n$INSTDIR"
    Abort
  ${EndIf}

  ClearErrors
  FileOpen $1 "$INSTDIR\.wiser_write_test.tmp" w
  ${If} ${Errors}
    MessageBox MB_ICONSTOP|MB_TOPMOST "The selected folder is not writable:$\r$\n$INSTDIR"
    Abort
  ${EndIf}
  FileWrite $1 "ok"
  FileClose $1
  Delete "$INSTDIR\.wiser_write_test.tmp"
FunctionEnd

Section "Install"
  ; In silent/elevated relaunch mode, normalize again in case /INSTALLDIR was passed as base path.
  Call NormalizeInstallDir

  ${If} $InstallScope == "all"
    SetShellVarContext all
  ${Else}
    SetShellVarContext current
  ${EndIf}

  ; Warn before continuing if this exact WISER version is already registered.
  ${If} $InstallScope == "all"
    ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "DisplayName"
    ReadRegStr $1 HKLM "${REGKEY_UNINSTALL}" "Publisher"
  ${Else}
    ReadRegStr $0 HKCU "${REGKEY_UNINSTALL}" "DisplayName"
    ReadRegStr $1 HKCU "${REGKEY_UNINSTALL}" "Publisher"
  ${EndIf}
  ${If} "$0" == "${APP_DIRNAME}"
    ${AndIf} "$1" == "California Institute of Technology"
    MessageBox MB_ICONEXCLAMATION|MB_OKCANCEL|MB_DEFBUTTON2 \
      "${APP_DIRNAME} is already installed. Installing without first uninstalling will mess with application registry keys.$\r$\n$\r$\nPress OK to continue with this installation or Cancel to stop." \
      IDOK +2 IDCANCEL 0
    Quit
  ${EndIf}

  ; If the current target directory already contains a WISER uninstaller, run it first.
  IfFileExists "$INSTDIR\${UNINSTALL_EXE_NAME}" 0 finish_uninstall_jump
    ClearErrors
    ExecWait '"$INSTDIR\${UNINSTALL_EXE_NAME}" /S' $1

    ${If} $1 != 0
      MessageBox MB_ICONSTOP|MB_TOPMOST "Previous/current dir uninstaller failed with exit code $1."
      Abort
    ${EndIf}

  finish_uninstall_jump:

  ; Delete previous install tree only when install marker exists.
  ; Never recursively delete arbitrary user-selected directories.
  IfFileExists "$INSTDIR\WISER.exe" 0 +3
    RMDIR /r "$INSTDIR"
    Goto +1

  SetOutPath "$INSTDIR"

  File /r dist\WISER\*.*
  File icons\wiser.ico

  ; Create uninstaller
  ; WriteUninstaller "$INSTDIR\Uninstall WISER.exe"
  !ifndef INNER
  File "$%TEMP%\${UNINSTALL_EXE_NAME}"
  !endif

  ; Create shortcuts to run and uninstall application
  CreateDirectory "$SMPROGRAMS\${STARTMENU_DIRNAME}"
  CreateShortcut "$SMPROGRAMS\${STARTMENU_DIRNAME}\${STARTMENU_APP_LINK}" "$INSTDIR\WISER.exe"
  CreateShortcut "$SMPROGRAMS\${STARTMENU_DIRNAME}\${STARTMENU_UNINSTALL_LINK}" "$INSTDIR\${UNINSTALL_EXE_NAME}"

  ; Write registry keys to uninstall app through Windows system console

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  ${If} $InstallScope == "all"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayName" "${APP_DIRNAME}"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "Publisher" "California Institute of Technology"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "RegCompany" "California Institute of Technology"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayVersion" "${WISER_VERSION}"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayIcon" "$\"$INSTDIR\wiser.ico$\""
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\${UNINSTALL_EXE_NAME}$\""
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\${UNINSTALL_EXE_NAME}$\" /S"
    WriteRegDWORD HKLM "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"
  ${Else}
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayName" "${APP_DIRNAME}"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "Publisher" "California Institute of Technology"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "RegCompany" "California Institute of Technology"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayVersion" "${WISER_VERSION}"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayIcon" "$\"$INSTDIR\wiser.ico$\""
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\${UNINSTALL_EXE_NAME}$\""
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\${UNINSTALL_EXE_NAME}$\" /S"
    WriteRegDWORD HKCU "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"
  ${EndIf}

SectionEnd

;--------------------------------
; Uninstaller Section

!ifdef INNER
Function un.onInit
  SetShellVarContext current
  ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "UninstallString"
  ${If} "$0" == "$\"$INSTDIR\${UNINSTALL_EXE_NAME}$\""
    SetShellVarContext all
  ${EndIf}
FunctionEnd

Section "Uninstall"

  ; Clean up the installed files.

  ; NOT NECESSARY? Delete "$INSTDIR\Uninstall WISER.exe"
  IfFileExists "$INSTDIR\WISER.exe" 0 +2
    RMDir /r "$INSTDIR"

  ; Clean up start-menu entries

  Delete "$SMPROGRAMS\${STARTMENU_DIRNAME}\${STARTMENU_APP_LINK}"
  Delete "$SMPROGRAMS\${STARTMENU_DIRNAME}\${STARTMENU_UNINSTALL_LINK}"
  RMDir /r "$SMPROGRAMS\${STARTMENU_DIRNAME}"

  ; Clean up registry keys

  ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "UninstallString"
  ${If} $0 != ""
    DeleteRegKey HKLM "${REGKEY_UNINSTALL}"
  ${Else}
    DeleteRegKey HKCU "${REGKEY_UNINSTALL}"
  ${EndIf}

SectionEnd
!endif
