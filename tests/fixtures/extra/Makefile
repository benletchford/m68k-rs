# Unified Makefile for Extra 68k Test Fixtures
#
# Directory structure:
#   m68040/src/  - 68040-specific tests
#   m68040/bin/  - Compiled 68040 binaries
#   m68020/src/  - 68020-specific tests
#   m68020/bin/  - Compiled 68020 binaries
#
# Build with Docker: ./build_fixtures.sh

M68K_AS = m68k-linux-gnu-as
M68K_LD = m68k-linux-gnu-ld
M68K_LDFLAGS = -Ttext 0x10000 --oformat binary

# Include path for entry.s
ASFLAGS_COMMON = -I../Musashi/test

# 68040-specific tests (includes EC040/LC040 variants)
M68040_FIXTURES = 32bit_disp scale_index mem_indirect pc_indirect \
	extb link32 move16 moves tas pack unpk \
	movec cache_ops pflush reset \
	bkpt illegal trace_modes exception_frames \
	multiprec rtd trapcc \
	fpu_arith fpu_move fpu_branch fpu_trans fpu_ctrl fpu_double fpu_unimp \
	fpu_transcendental fpu_transcendental2 fpu_exp_log \
	fpu_basic_ops fpu_sincos fpu_rounding fpu_constants fpu_remainder fpu_scale \
	mmu_ptest mmu_regs mmu_ttr mmu_atc mmu_tc mmu_permissions \
	mmu_table_walk mmu_atc_test mmu_ttr_config \
	movec_040 cycle_timing cycle_exception cycle_addressing bus_error \
	dasm_data_move dasm_arithmetic dasm_logical dasm_shift dasm_branch \
	interrupts fpu_except callm_rtm \
	smc_test double_fault address_stress \
	arch_unaligned arch_fpu_trap ec040_mmu_trap \
	lc040_mmu_positive ec040_positive exception_frame_format

# 68020-specific tests (includes EC020/EC030 variants)
M68020_FIXTURES = bitfield_ops cas chk2_cmp2 long_muldiv boundary_edge \
	msp_test trace_t0 callm_020 ec020_mmu_trap ec030_fpu ec030_mmu_trap

# 68030-specific tests
M68030_FIXTURES = mmu_030_tc cache_030 move16_030

# 68010-specific tests
M68010_FIXTURES = vbr_test move_ccr rte_010 loop_mode

# Privilege tests (all CPUs)
PRIVILEGE_FIXTURES = user_movec user_reset user_stop user_rte \
	usp_ssp_switch move_usp trap_privilege exception_privilege \
	rte_privilege privilege_boundary

# Exception nesting tests (all CPUs)
EXCEPTION_FIXTURES = double_exception exception_priority \
	interrupt_nesting rte_validation spurious_interrupt

# Systematic instruction coverage tests (all CPUs)
COVERAGE_FIXTURES = bset_bclr bchg_btst bit_edge_cases bit_memory_ea \
	asl_asr lsl_lsr rol_ror roxl_roxr shift_count_zero shift_count_large \
	shift_memory shift_flags \
	bcc_all dbcc_all scc_all trapcc condition_edge \
	mulu_edge muls_edge divu_divs_edge \
	cmpm_all movem_patterns movem_stack \
	exg_variants swap_ext

M68040_SOURCES = $(addprefix m68040/src/, $(M68040_FIXTURES:=.s))
M68040_BINARIES = $(addprefix m68040/bin/, $(M68040_FIXTURES:=.bin))

M68020_SOURCES = $(addprefix m68020/src/, $(M68020_FIXTURES:=.s))
M68020_BINARIES = $(addprefix m68020/bin/, $(M68020_FIXTURES:=.bin))

M68030_SOURCES = $(addprefix m68030/src/, $(M68030_FIXTURES:=.s))
M68030_BINARIES = $(addprefix m68030/bin/, $(M68030_FIXTURES:=.bin))

M68010_SOURCES = $(addprefix m68010/src/, $(M68010_FIXTURES:=.s))
M68010_BINARIES = $(addprefix m68010/bin/, $(M68010_FIXTURES:=.bin))

PRIVILEGE_SOURCES = $(addprefix privilege/src/, $(PRIVILEGE_FIXTURES:=.s))
PRIVILEGE_BINARIES = $(addprefix privilege/bin/, $(PRIVILEGE_FIXTURES:=.bin))

EXCEPTION_SOURCES = $(addprefix exceptions/src/, $(EXCEPTION_FIXTURES:=.s))
EXCEPTION_BINARIES = $(addprefix exceptions/bin/, $(EXCEPTION_FIXTURES:=.bin))

COVERAGE_SOURCES = $(addprefix coverage/src/, $(COVERAGE_FIXTURES:=.s))
COVERAGE_BINARIES = $(addprefix coverage/bin/, $(COVERAGE_FIXTURES:=.bin))

.PHONY: all clean list m68040 m68020 m68030 m68010 privilege exceptions coverage

all: m68040 m68020 m68030 m68010 privilege exceptions coverage

m68040: $(M68040_BINARIES)

m68020: $(M68020_BINARIES)

m68030: $(M68030_BINARIES)

m68010: $(M68010_BINARIES)

privilege: $(PRIVILEGE_BINARIES)

exceptions: $(EXCEPTION_BINARIES)

coverage: $(COVERAGE_BINARIES)

# 68040 build rules
m68040/bin/%.bin: m68040/src/%.s | m68040/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68040 $(ASFLAGS_COMMON) -o m68040/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ m68040/bin/$*.o
	@rm -f m68040/bin/$*.o

# 68020 build rules
m68020/bin/%.bin: m68020/src/%.s | m68020/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68020 $(ASFLAGS_COMMON) -o m68020/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ m68020/bin/$*.o
	@rm -f m68020/bin/$*.o

# 68030 build rules
m68030/bin/%.bin: m68030/src/%.s | m68030/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68030 $(ASFLAGS_COMMON) -o m68030/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ m68030/bin/$*.o
	@rm -f m68030/bin/$*.o

# 68010 build rules
m68010/bin/%.bin: m68010/src/%.s | m68010/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68010 $(ASFLAGS_COMMON) -o m68010/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ m68010/bin/$*.o
	@rm -f m68010/bin/$*.o

#Privilege build rules (use 68040 for maximum compatibility)
privilege/bin/%.bin: privilege/src/%.s | privilege/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68040 $(ASFLAGS_COMMON) -o privilege/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ privilege/bin/$*.o
	@rm -f privilege/bin/$*.o

# Exception build rules (use 68040 for maximum compatibility)
exceptions/bin/%.bin: exceptions/src/%.s | exceptions/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68040 $(ASFLAGS_COMMON) -o exceptions/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ exceptions/bin/$*.o
	@rm -f exceptions/bin/$*.o

# Coverage build rules (use 68040 for maximum compatibility)
coverage/bin/%.bin: coverage/src/%.s | coverage/bin
	@echo "Building $< -> $@"
	@$(M68K_AS) -m68040 $(ASFLAGS_COMMON) -o coverage/bin/$*.o $<
	@$(M68K_LD) $(M68K_LDFLAGS) -o $@ coverage/bin/$*.o
	@rm -f coverage/bin/$*.o

m68040/bin:
	mkdir -p m68040/bin

m68020/bin:
	mkdir -p m68020/bin

m68030/bin:
	mkdir -p m68030/bin

m68010/bin:
	mkdir -p m68010/bin

privilege/bin:
	mkdir -p privilege/bin

exceptions/bin:
	mkdir -p exceptions/bin

coverage/bin:
	mkdir -p coverage/bin

clean:
	rm -f m68040/bin/*.bin m68040/bin/*.o
	rm -f m68020/bin/*.bin m68020/bin/*.o
	rm -f m68030/bin/*.bin m68030/bin/*.o
	rm -f m68010/bin/*.bin m68010/bin/*.o
	rm -f privilege/bin/*.bin privilege/bin/*.o
	rm -f exceptions/bin/*.bin exceptions/bin/*.o
	rm -f coverage/bin/*.bin coverage/bin/*.o

list:
	@echo "68040 Fixtures ($(words $(M68040_FIXTURES))):"
	@echo "$(M68040_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "68020 Fixtures ($(words $(M68020_FIXTURES))):"
	@echo "$(M68020_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "68030 Fixtures ($(words $(M68030_FIXTURES))):"
	@echo "$(M68030_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "68010 Fixtures ($(words $(M68010_FIXTURES))):"
	@echo "$(M68010_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "Privilege Fixtures ($(words $(PRIVILEGE_FIXTURES))):"
	@echo "$(PRIVILEGE_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "Exception Fixtures ($(words $(EXCEPTION_FIXTURES))):"
	@echo "$(EXCEPTION_FIXTURES)" | tr ' ' '\n'
	@echo ""
	@echo "Coverage Fixtures ($(words $(COVERAGE_FIXTURES))):"
	@echo "$(COVERAGE_FIXTURES)" | tr ' ' '\n'
