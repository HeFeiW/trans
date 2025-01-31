#include <EGL/egl.h>
#include <stdio.h>

int main() {
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        printf("Failed to get EGL display\n");
        return -1;
    }
    
    if (!eglInitialize(display, NULL, NULL)) {
        printf("Failed to initialize EGL\n");
        return -1;
    }
    
    printf("EGL initialized successfully!\n");
    eglTerminate(display);
    return 0;
}