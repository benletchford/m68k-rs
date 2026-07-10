/* Flat 16MB big-endian memory for the Musashi core.
 *
 * Mirrors the Rust side's benchmark bus: every access is mask-and-index into
 * one contiguous buffer, so both emulators pay the same (minimal) memory-system
 * cost and the comparison isolates the CPU cores themselves. Musashi always
 * calls through these extern functions; that call-per-access interface is how
 * Musashi is deployed in practice (e.g. in MAME), so it is not handicapped
 * relative to real-world use.
 */

#include "m68k.h"

#define MEM_SIZE 0x1000000u
#define MEM_MASK (MEM_SIZE - 1u)

static unsigned char g_mem[MEM_SIZE];

unsigned char *musashi_mem_ptr(void)
{
	return g_mem;
}

unsigned int m68k_read_memory_8(unsigned int address)
{
	return g_mem[address & MEM_MASK];
}

unsigned int m68k_read_memory_16(unsigned int address)
{
	unsigned int a = address & MEM_MASK;
	return ((unsigned int)g_mem[a] << 8) | g_mem[(a + 1u) & MEM_MASK];
}

unsigned int m68k_read_memory_32(unsigned int address)
{
	unsigned int a = address & MEM_MASK;
	return ((unsigned int)g_mem[a] << 24)
	     | ((unsigned int)g_mem[(a + 1u) & MEM_MASK] << 16)
	     | ((unsigned int)g_mem[(a + 2u) & MEM_MASK] << 8)
	     | (unsigned int)g_mem[(a + 3u) & MEM_MASK];
}

void m68k_write_memory_8(unsigned int address, unsigned int value)
{
	g_mem[address & MEM_MASK] = (unsigned char)value;
}

void m68k_write_memory_16(unsigned int address, unsigned int value)
{
	unsigned int a = address & MEM_MASK;
	g_mem[a] = (unsigned char)(value >> 8);
	g_mem[(a + 1u) & MEM_MASK] = (unsigned char)value;
}

void m68k_write_memory_32(unsigned int address, unsigned int value)
{
	unsigned int a = address & MEM_MASK;
	g_mem[a] = (unsigned char)(value >> 24);
	g_mem[(a + 1u) & MEM_MASK] = (unsigned char)(value >> 16);
	g_mem[(a + 2u) & MEM_MASK] = (unsigned char)(value >> 8);
	g_mem[(a + 3u) & MEM_MASK] = (unsigned char)value;
}
