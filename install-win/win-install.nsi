;--------------------------------
; Headers

!include "MUI2.nsh"
!include "FileFunc.nsh"

;--------------------------------
; Project Settings

Name "Workbench for Imaging Spectroscopy Exploration and Research"
; TODO(donnie):  Get version from external file.
OutFile "Install-WISER-1.0a3.exe"
Unicode True
ManifestDPIAware True

; TODO(donnie):  Currently we build a 64-bit Python frozen app.
InstallDir "$PROGRAMFILES64\WISER"

!define REGKEY_UNINSTALL "Software\Microsoft\Windows\CurrentVersion\Uninstall\WISER"

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

  File /r dist\WISER\*.*

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Create shortcuts to run and uninstall application
  CreateDirectory "$SMPROGRAMS\WISER"
  CreateShortcut "$SMPROGRAMS\WISER\WISER.lnk" "$INSTDIR\WISER.exe"
  CreateShortcut "$SMPROGRAMS\WISER\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

  ; Write registry keys to uninstall app through Windows system console

  WriteRegStr HKLM "${REGKEY_UNINSTALL}" "DisplayName" "WISER"
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

  Delete "$SMPROGRAMS\WISER\WISER.lnk"
  Delete "$SMPROGRAMS\WISER\Uninstall.lnk"
  RMDir /r "$SMPROGRAMS\WISER"

  ; Clean up registry keys

  DeleteRegKey HKLM "${REGKEY_UNINSTALL}"

SectionEnd
