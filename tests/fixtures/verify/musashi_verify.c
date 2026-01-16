/*
 * Musashi fixture verifier - runs coverage test binaries through Musashi
 * to verify that original (buggy) fixtures fail on real 68k emulation.
 *
 * Usage: musashi_verify <fixture.bin>
 *
 * Test device protocol:
 * - Write 1 to 0xA00000 = PASS
 * - Write 1 to 0xA00004 = FAIL
 * - STOP instruction = test complete
 */

#include "../m68k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Memory layout matching our test fixtures */
#define ROM_BASE 0x10000
#define ROM_SIZE 0x10000
#define RAM_BASE 0x0
#define RAM_SIZE 0x10000
/* Musashi test protocol addresses from entry.s */
#define TEST_FAIL_ADDR 0x100000
#define TEST_PASS_ADDR 0x100004

/* Memory */
static unsigned char g_rom[ROM_SIZE];
static unsigned char g_ram[RAM_SIZE];

/* Test result */
static int g_pass_count = 0;
static int g_fail_count = 0;
static int g_stopped = 0;

/* Read macros */
#define READ_BYTE(BASE, ADDR) (BASE)[ADDR]
#define READ_WORD(BASE, ADDR) (((BASE)[ADDR] << 8) | (BASE)[(ADDR) + 1])
#define READ_LONG(BASE, ADDR)                                                  \
  (((BASE)[ADDR] << 24) | ((BASE)[(ADDR) + 1] << 16) |                         \
   ((BASE)[(ADDR) + 2] << 8) | (BASE)[(ADDR) + 3])

#define WRITE_BYTE(BASE, ADDR, VAL) (BASE)[ADDR] = (VAL) & 0xff
#define WRITE_WORD(BASE, ADDR, VAL)                                            \
  do {                                                                         \
    (BASE)[ADDR] = ((VAL) >> 8) & 0xff;                                        \
    (BASE)[(ADDR) + 1] = (VAL) & 0xff;                                         \
  } while (0)
#define WRITE_LONG(BASE, ADDR, VAL)                                            \
  do {                                                                         \
    (BASE)[ADDR] = ((VAL) >> 24) & 0xff;                                       \
    (BASE)[(ADDR) + 1] = ((VAL) >> 16) & 0xff;                                 \
    (BASE)[(ADDR) + 2] = ((VAL) >> 8) & 0xff;                                  \
    (BASE)[(ADDR) + 3] = (VAL) & 0xff;                                         \
  } while (0)

/* Musashi memory callbacks */
unsigned int m68k_read_memory_8(unsigned int address) {
  if (address >= ROM_BASE && address < ROM_BASE + ROM_SIZE)
    return READ_BYTE(g_rom, address - ROM_BASE);
  if (address < RAM_SIZE)
    return READ_BYTE(g_ram, address);
  return 0;
}

unsigned int m68k_read_memory_16(unsigned int address) {
  if (address >= ROM_BASE && address < ROM_BASE + ROM_SIZE)
    return READ_WORD(g_rom, address - ROM_BASE);
  if (address < RAM_SIZE)
    return READ_WORD(g_ram, address);
  return 0;
}

unsigned int m68k_read_memory_32(unsigned int address) {
  if (address >= ROM_BASE && address < ROM_BASE + ROM_SIZE)
    return READ_LONG(g_rom, address - ROM_BASE);
  if (address < RAM_SIZE)
    return READ_LONG(g_ram, address);
  return 0;
}

void m68k_write_memory_8(unsigned int address, unsigned int value) {
  if (address == TEST_PASS_ADDR) {
    g_pass_count++;
    return;
  }
  if (address == TEST_FAIL_ADDR) {
    g_fail_count++;
    return;
  }
  if (address < RAM_SIZE)
    WRITE_BYTE(g_ram, address, value);
}

void m68k_write_memory_16(unsigned int address, unsigned int value) {
  if (address == TEST_PASS_ADDR) {
    g_pass_count++;
    return;
  }
  if (address == TEST_FAIL_ADDR) {
    g_fail_count++;
    return;
  }
  if (address < RAM_SIZE)
    WRITE_WORD(g_ram, address, value);
}

void m68k_write_memory_32(unsigned int address, unsigned int value) {
  if (address >= 0x100000) {
    fprintf(stderr, "  WRITE32: addr=%08x val=%08x\n", address, value);
  }
  if (address == TEST_PASS_ADDR) {
    g_pass_count++;
    return;
  }
  if (address == TEST_FAIL_ADDR) {
    g_fail_count++;
    return;
  }
  if (address < RAM_SIZE)
    WRITE_LONG(g_ram, address, value);
}

unsigned int m68k_read_disassembler_16(unsigned int address) {
  return m68k_read_memory_16(address);
}

unsigned int m68k_read_disassembler_32(unsigned int address) {
  return m68k_read_memory_32(address);
}

/* Instruction callback to detect STOP */
void m68k_instr_callback(int pc) {
  unsigned int opcode = m68k_read_memory_16(pc);
  if ((opcode & 0xFFF8) == 0x4E70 && (opcode & 0x0007) == 0x0002) {
    /* STOP instruction */
    g_stopped = 1;
  }
}

int main(int argc, char *argv[]) {
  FILE *f;
  size_t size;
  int cycles;

  if (argc != 2) {
    fprintf(stderr, "Usage: %s <fixture.bin>\n", argv[0]);
    return 1;
  }

  /* Load fixture */
  f = fopen(argv[1], "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", argv[1]);
    return 1;
  }

  size = fread(g_rom, 1, ROM_SIZE, f);
  fclose(f);

  if (size <= 0) {
    fprintf(stderr, "Empty file\n");
    return 1;
  }

  /* Setup initial stack and PC in vector table (RAM) */
  /* The fixture has entry point at ROM_BASE (0x10000) */
  WRITE_LONG(g_ram, 0, 0x3F0);    /* Initial SP */
  WRITE_LONG(g_ram, 4, ROM_BASE); /* Initial PC */

  /* Initialize Musashi */
  m68k_init();
  m68k_set_cpu_type(M68K_CPU_TYPE_68040);
  m68k_pulse_reset();

  /* Debug: print initial state */
  fprintf(stderr, "Initial PC: %08x, SP: %08x\n",
          m68k_get_reg(NULL, M68K_REG_PC), m68k_get_reg(NULL, M68K_REG_A7));

  /* Run test */
  cycles = 0;
  while (!g_stopped && cycles < 100000) {
    int step_cycles = m68k_execute(1);
    cycles += step_cycles;

    unsigned int pc = m68k_get_reg(NULL, M68K_REG_PC);
    unsigned int opcode = m68k_read_memory_16(pc);

    /* Check for STOP instruction (0x4E72) */
    if (opcode == 0x4E72) {
      g_stopped = 1;
      break;
    }

    /* Debug first 10 cycles */
    if (cycles <= 100) {
      fprintf(stderr, "  cycles=%d PC=%08x op=%04x\n", cycles, pc, opcode);
    }
  }

  printf("Test %s: passes=%d, fails=%d\n", argv[1], g_pass_count, g_fail_count);

  if (g_fail_count > 0)
    return 1;
  if (g_pass_count == 0)
    return 2;
  return 0;
}
