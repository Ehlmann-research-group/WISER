;============================================================================
; To code-sign the uninstaller, this file makes use of this technique:
; https://nsis.sourceforge.io/Signing_an_Uninstaller

; !verbose 4

;--------------------------------
; Headers

!include "MUI2.nsh"
!include "FileFunc.nsh"

;--------------------------------
; Project Settings

Name "Workbench for Imaging Spectroscopy Exploration and Research"
Unicode True
ManifestDPIAware True

; TODO(donnie):  Currently we build a 64-bit Python frozen app.
InstallDir "$PROGRAMFILES64\WISER"

!define REGKEY_UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\WISER"


!ifdef INNER
  !echo "Inner invocation"                  ; just to see what's going on
  OutFile "$%TEMP%\tempinstaller.exe"       ; not really important where this is
  SetCompress off                           ; for speed
!else
  !echo "Outer invocation"

  ; Call makensis again against current file, defining INNER.  This writes an installer for us which, when
  ; it is invoked, will just write the uninstaller to some location, and then exit.

  !makensis '/NOCD /DINNER "install-win\win-install.nsi"' = 0

  ; So now run that installer we just created as %TEMP%\tempinstaller.exe.  Since it
  ; calls quit the return value isn't zero.

  !system 'set __COMPAT_LAYER=RunAsInvoker&"$%TEMP%\tempinstaller.exe"' = 2

  ; That will have written an uninstaller binary for us.  Now we sign it with your
  ; favorite code signing tool.

  !system '"C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool" sign /f C:\Users\donnie\WISER-CodeSign.pfx /p WISER /t http://timestamp.sectigo.com "%TEMP%\Uninstall WISER.exe"' = 0

  ; Good.  Now we can carry on writing the real installer.

  ; TODO(donnie):  Get version from external file.
  OutFile "Install-WISER-1.1a4-dev0.exe"
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

; ...[the rest of your normal .onInit]...
FunctionEnd

!ifndef INNER
!finalize '"C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool" sign /f C:\Users\donnie\WISER-CodeSign.pfx /p WISER /t http://timestamp.sectigo.com "%1"' = 0
!endif

;--------------------------------
; Modern UI 2 Specification

; Installer

!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "install-win\license.rtf"
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

Section "Install"

  SetOutPath "$INSTDIR"

  File /r dist\WISER\*.*

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

  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayName" "WISER"
  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\""

  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\Uninstall WISER.exe$\" /S"

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"

SectionEnd

;--------------------------------
; Uninstaller Section

!ifdef INNER
Section "Uninstall"

  ; Clean up the installed files.

  ; NOT NECESSARY? Delete "$INSTDIR\Uninstall WISER.exe"
  RMDir /r "$INSTDIR"

  ; Clean up start-menu entries

  Delete "$SMPROGRAMS\WISER\WISER.lnk"
  Delete "$SMPROGRAMS\WISER\Uninstall WISER.lnk"
  RMDir /r "$SMPROGRAMS\WISER"

  ; Clean up registry keys

  DeleteRegKey HKLM "${REGKEY_UNINSTALL}"

SectionEnd
!endif
