#include <am.h>
#include <amdev.h>
#include <klib.h>
#include <klib-macros.h>

void splash();
//void print_key();
int32_t print_key();
void update_screen();
static inline void puts(const char *s) {
  for (; *s; s++) putch(*s);
}
