;--------------------------------
; Headers

!include "MUI2.nsh"
!include "FileFunc.nsh"

;--------------------------------
; Project Settings

Name "Imaging Spectroscopy Workbench"
; TODO(donnie):  Get version from external file.
OutFile "Install-ISWB-0.0.1.exe"
Unicode True
ManifestDPIAware True

; TODO(donnie):  Currently we build a 64-bit Python frozen app.
InstallDir "$PROGRAMFILES64\Imaging Spectroscopy Workbench"

!define REGKEY_UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\ISWB"

;--------------------------------
; Modern UI 2 Specification

; Installer

!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "install-win\license.rtf"
!insertmacro MUI_PAGE_INSTFILES

; Uninstaller

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Language

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Section

Section "Install"

  SetOutPath "$INSTDIR"

  File /r dist\ISWorkbench\*.*

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Create shortcuts to run and uninstall application
  CreateDirectory "$SMPROGRAMS\ISWB"
  CreateShortcut "$SMPROGRAMS\ISWB\ISWorkbench.lnk" "$INSTDIR\ISWorkbench.exe"
  CreateShortcut "$SMPROGRAMS\ISWB\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

  ; Write registry keys to uninstall app through Windows system console

  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayName" "Imaging Spectroscopy Workbench"
  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""

  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "QuietUninstallString" "$\"$INSTDIR\Uninstall.exe$\" /S"

  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${REGKEY_UNINSTALL}" "EstimatedSize" "$0"

SectionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"

  ; Clean up the installed files.

  ; NOT NECESSARY? Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"

  ; Clean up start-menu entries

  Delete "$SMPROGRAMS\ISWB\ISWorkbench.lnk"
  Delete "$SMPROGRAMS\ISWB\Uninstall.lnk"
  RMDir /r "$SMPROGRAMS\ISWB"

  ; Clean up registry keys

  DeleteRegKey HKLM "${REGKEY_UNINSTALL}"

SectionEnd
