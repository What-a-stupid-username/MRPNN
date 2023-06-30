#pragma once


#if LINUX

#include "unistd.h"

#define wait usleep

#define OpenBMP(pic) ;

#else


#include <Windows.h>


#define wait Sleep


#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#define OpenBMP(pic) ShellExecuteA(NULL, "open", (LPCSTR)((pic) + ".bmp").c_str(), NULL, NULL, 1)

#endif