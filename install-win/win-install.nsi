;--------------------------------
; Headers

!include "MUI2.nsh"

;--------------------------------
; Project Settings

Name "Imaging Spectroscopy Workbench"
; TODO(donnie):  Get version from external file.
OutFile "Install-ISWB-0.0.1.exe"
Unicode True
ManifestDPIAware True

; TODO(donnie):  Currently we build a 64-bit Python frozen app.
InstallDir "$PROGRAMFILES64\Imaging Spectroscopy Workbench"

;--------------------------------
; Modern UI 2 Specification

; Installer

!define MUI_LICENSEPAGE_CHECKBOX
!insertmacro MUI_PAGE_LICENSE "license.rtf"
!insertmacro MUI_PAGE_INSTFILES

; Uninstaller

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Language

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Section

Section "Install"

  SetOutPath $INSTDIR

  File /r dist\ISWorkbench\*.*

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Create shortcuts to run and uninstall application
  CreateDirectory "$SMPROGRAMS\ISWB"
  CreateShortcut "$SMPROGRAMS\ISWB\ISWorkbench.lnk" "$INSTDIR\ISWorkbench.exe"
  CreateShortcut "$SMPROGRAMS\ISWB\Uninstall.lnk" "$INSTDIR\Uninstall.exe"

  ; Write registry keys to uninstall app through Windows system console

  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ISWB" \
                   "DisplayName" "Imaging Spectroscopy Workbench"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ISWB" \
                   "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""

SectionEnd

;--------------------------------
; Uninstaller Section

Section "Uninstall"

  ; Clean up the installed files.

  ; NOT NECESSARY? Delete "$INSTDIR\Uninstall.exe"
  RMDir /r $INSTDIR

  ; Clean up start-menu entries

  Delete "$SMPROGRAMS\ISWB\ISWorkbench.lnk"
  Delete "$SMPROGRAMS\ISWB\Uninstall.lnk"
  RMDir /r "$SMPROGRAMS\ISWB"

  ; Clean up registry keys

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ISWB"

SectionEnd
