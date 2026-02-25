!macro customHeader
  !system "echo 'Custom NSIS header loaded'"
!macroend

!macro preInit
  FileOpen $9 "$DESKTOP\imagine-install.log" w
  FileWrite $9 "=== Imagine Installer Log ===$\r$\n"
  FileWrite $9 "preInit: OK$\r$\n"
  FileClose $9
!macroend

!macro customInit
  FileOpen $9 "$DESKTOP\imagine-install.log" a
  FileSeek $9 0 END
  FileWrite $9 "customInit: starting$\r$\n"
  FileWrite $9 "INSTDIR: $INSTDIR$\r$\n"
  FileWrite $9 "customInit: OK$\r$\n"
  FileClose $9
!macroend

!macro customInstall
  FileOpen $9 "$DESKTOP\imagine-install.log" a
  FileSeek $9 0 END
  FileWrite $9 "customInstall: extraction complete$\r$\n"

  IfFileExists "$INSTDIR\Imagine.exe" 0 +3
    FileWrite $9 "  Imagine.exe: found$\r$\n"
    Goto +2
    FileWrite $9 "  Imagine.exe: NOT found$\r$\n"

  IfFileExists "$INSTDIR\resources\app.asar" 0 +3
    FileWrite $9 "  app.asar: found$\r$\n"
    Goto +2
    FileWrite $9 "  app.asar: NOT found$\r$\n"

  FileWrite $9 "customInstall: OK$\r$\n"
  FileClose $9
!macroend

!macro customUnInit
!macroend
