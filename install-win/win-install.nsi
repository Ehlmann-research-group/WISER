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

; --- Per-user defaults (no admin) ---
RequestExecutionLevel user

InstallDir "$LOCALAPPDATA"

!define REGKEY_UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\WISER"

Var InstallScope
Var RelaunchArgs


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

  !system '"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%TEMP%\Uninstall WISER.exe"' = 0

  ; Good.  Now we can carry on writing the real installer.

  OutFile "Install-WISER-${WISER_VERSION}.exe"
  ; SetCompressor /SOLID lzma
!endif


 Function .onInit
!ifdef INNER

  ; If INNER is defined, then we aren't supposed to do anything except write out
  ; the uninstaller.  This is better than processing a command line option as it means
  ; this entire code path is not present in the final (real) installer.
  SetSilent silent
  WriteUninstaller "$%TEMP%\Uninstall WISER.exe"
  Quit  ; just bail out quickly when running the "inner" installer
!endif

  StrCpy $InstallScope "current"
  SetShellVarContext current

  ${GetParameters} $0
  ${GetOptions} $0 "/ALLUSERS=" $1
  ${If} $1 == "1"
    StrCpy $InstallScope "all"
    SetShellVarContext all
    SetSilent silent
  ${EndIf}

  ${GetOptions} $0 "/INSTALLDIR=" $2
  ${If} $2 != ""
    StrCpy $INSTDIR $2
  ${EndIf}
FunctionEnd

!ifndef INNER
!finalize '"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool" sign /sha1 "${SHA1_THUMBPRINT}" /fd SHA256 /t http://timestamp.sectigo.com "%1"' = 0
!endif

;--------------------------------
; Modern UI 2 Specification

; Installer

!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "install-win\license.rtf"
!ifndef INNER
!define MUI_PAGE_CUSTOMFUNCTION_LEAVE ValidateInstallDir
!insertmacro MUI_PAGE_DIRECTORY
!undef MUI_PAGE_CUSTOMFUNCTION_LEAVE
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
  ${If} "$0" != "WISER"
    ${AndIf} "$0" != "wiser"
    StrCpy $INSTDIR "$INSTDIR\WISER"
  ${EndIf}
FunctionEnd

Function LaunchElevatedAllUsers
  StrCpy $RelaunchArgs '/ALLUSERS=1 /INSTALLDIR="$INSTDIR" /S'
  ExecShell "runas" "$EXEPATH" $RelaunchArgs
  ${If} ${Errors}
    MessageBox MB_ICONSTOP|MB_TOPMOST "Unable to relaunch installer as administrator."
    Abort
  ${EndIf}
  Quit
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

  CreateDirectory "$INSTDIR"
  ${If} ${Errors}
    MessageBox MB_ICONSTOP|MB_TOPMOST "Unable to create install directory:$\r$\n$INSTDIR"
    Abort
  ${EndIf}

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

  ; Check to see if the application already exists
  ; If so, we run the uninstaller in the selected scope only.
  ${If} $InstallScope == "all"
    ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "UninstallString"
  ${Else}
    ReadRegStr $0 HKCU "${REGKEY_UNINSTALL}" "UninstallString"
  ${EndIf}
  StrCmp $0 "" +1
  ExecWait '"$0"'

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
  File "$%TEMP%\Uninstall WISER.exe"
  !endif

  ; Create shortcuts to run and uninstall application
  CreateDirectory "$SMPROGRAMS\WISER"
  CreateShortcut "$SMPROGRAMS\WISER\WISER.lnk" "$INSTDIR\WISER.exe"
  CreateShortcut "$SMPROGRAMS\WISER\Uninstall WISER.lnk" "$INSTDIR\Uninstall WISER.exe"

  ; Write registry keys to uninstall app through Windows system console

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  ${If} $InstallScope == "all"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayName" "WISER"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "Publisher" "California Institute of Technology"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "RegCompany" "California Institute of Technology"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayVersion" "${WISER_VERSION}"
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayIcon" "$\"$INSTDIR\wiser.ico$\""
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\""
    WriteRegStr HKLM "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\" /S"
    WriteRegDWORD HKLM "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"
  ${Else}
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayName" "WISER"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "Publisher" "California Institute of Technology"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "RegCompany" "California Institute of Technology"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayVersion" "${WISER_VERSION}"
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "DisplayIcon" "$\"$INSTDIR\wiser.ico$\""
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\""
    WriteRegStr HKCU "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\" /S"
    WriteRegDWORD HKCU "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"
  ${EndIf}

SectionEnd

;--------------------------------
; Uninstaller Section

!ifdef INNER
Function un.onInit
  SetShellVarContext current
  ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "UninstallString"
  ${If} "$0" == "$\"$INSTDIR\Uninstall WISER.exe$\""
    SetShellVarContext all
  ${EndIf}
FunctionEnd

Section "Uninstall"

  ; Clean up the installed files.

  ; NOT NECESSARY? Delete "$INSTDIR\Uninstall WISER.exe"
  IfFileExists "$INSTDIR\WISER.exe" 0 +2
    RMDir /r "$INSTDIR"

  ; Clean up start-menu entries

  Delete "$SMPROGRAMS\WISER\WISER.lnk"
  Delete "$SMPROGRAMS\WISER\Uninstall WISER.lnk"
  RMDir /r "$SMPROGRAMS\WISER"

  ; Clean up registry keys

  ReadRegStr $0 HKLM "${REGKEY_UNINSTALL}" "UninstallString"
  ${If} $0 != ""
    DeleteRegKey HKLM "${REGKEY_UNINSTALL}"
  ${Else}
    DeleteRegKey HKCU "${REGKEY_UNINSTALL}"
  ${EndIf}

SectionEnd
!endif
