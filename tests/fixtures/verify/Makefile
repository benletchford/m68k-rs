# Makefile for Musashi fixture verifier

# MUSASHI_DIR can be overridden, defaults to sibling directory
MUSASHI_DIR ?= ../Musashi
MUSASHI_SRC = $(MUSASHI_DIR)/m68kcpu.c $(MUSASHI_DIR)/m68kdasm.c $(MUSASHI_DIR)/softfloat/softfloat.c

# Generate m68kops.c and m68kops.h first
M68KMAKE = $(MUSASHI_DIR)/m68kmake
M68K_IN = $(MUSASHI_DIR)/m68k_in.c

all: musashi_verify

m68kmake: $(MUSASHI_DIR)/m68kmake.c
	$(CC) -o $@ $<

m68kops.c m68kops.h: m68kmake $(M68K_IN)
	./m68kmake . $(M68K_IN)

musashi_verify: musashi_verify.c m68kops.c $(MUSASHI_SRC)
	$(CC) -O2 -I$(MUSASHI_DIR) -I$(MUSASHI_DIR)/softfloat -I. \
		-o $@ musashi_verify.c m68kops.c $(MUSASHI_SRC) \
		-DMUSASHI_CNF=\"m68kconf.h\" -lm

clean:
	rm -f musashi_verify m68kmake m68kops.c m68kops.h

.PHONY: all clean
