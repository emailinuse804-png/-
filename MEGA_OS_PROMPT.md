# MEGA OS GENERATION PROMPT

> Copy everything below the line and paste it into the most powerful AI model you have access to.
> This prompt is designed to be fed to frontier-class models (GPT-4+, Claude Opus, Gemini Ultra, etc.)
> For best results, feed it in stages (Phase 1 first, then Phase 2, etc.) so the model can focus deeply on each layer.

---

## THE PROMPT BEGINS HERE

---

You are an elite operating-system architect, kernel engineer, graphics-stack developer, networking specialist, security researcher, and systems programmer — all rolled into one. Your mission is to design and implement **AuroraOS** — a fully functional, modern, general-purpose operating system from absolute scratch. AuroraOS must be capable of:

1. **Running graphical games** (2D and 3D via a custom or ported GPU driver stack)
2. **Running Ollama** (local LLM inference engine) natively
3. **Running a full web browser** (capable of rendering modern HTML5/CSS3/JS websites)
4. **General-purpose desktop computing** (file management, text editing, terminal, media playback)

You will produce **complete, compilable, working source code** — not pseudocode, not summaries, not architecture diagrams alone. Every file, every header, every Makefile. Where a full implementation would exceed output limits, you will produce the maximal working skeleton with clearly marked `// TODO: expand` stubs that compile and can be incrementally filled in, plus exact instructions for what each stub must do.

---

# ============================================================
# PHASE 1 — BOOTLOADER & KERNEL CORE
# ============================================================

## 1.1 Bootloader (x86_64, UEFI + Legacy BIOS fallback)

Produce a two-stage bootloader:

### Stage 1 — BIOS/UEFI Entry
- For BIOS: a 512-byte MBR boot sector in NASM x86 assembly that:
  - Sets up a minimal GDT (code/data segments, flat model)
  - Enables A20 line
  - Loads Stage 2 from disk (LBA read via INT 13h extensions)
  - Jumps to Stage 2

- For UEFI: an EFI application (written in C using gnu-efi or POSIX-UEFI) that:
  - Obtains the memory map via `GetMemoryMap()`
  - Loads the kernel ELF from an ESP FAT32 partition
  - Sets up a linear framebuffer via `GOP` (Graphics Output Protocol)
  - Passes a boot-info struct to the kernel (memory map, framebuffer address, size, pitch, pixel format)
  - Calls `ExitBootServices()` and jumps to the kernel entry point

### Stage 2 (BIOS path)
- Written in a mix of NASM assembly and C (with a minimal freestanding C runtime)
- Switches to Protected Mode, then Long Mode (64-bit)
- Sets up initial page tables (identity-map first 4 GiB, higher-half kernel mapping at `0xFFFFFFFF80000000`)
- Detects available memory via E820
- Loads the kernel ELF image from disk (parse ELF64 headers, load PT_LOAD segments)
- Populates a boot-info struct identical to the UEFI path
- Jumps to the kernel entry point in 64-bit mode

### Deliverables:
```
boot/
├── bios/
│   ├── stage1.asm          # MBR boot sector
│   ├── stage2.asm          # Protected/Long mode setup
│   ├── stage2_c.c          # C portion of stage 2 (ELF loader, memory detection)
│   ├── stage2_c.h
│   └── linker_stage2.ld
├── uefi/
│   ├── main.c              # UEFI application entry
│   ├── elf_loader.c        # ELF parsing and loading
│   ├── elf_loader.h
│   ├── gop.c               # Graphics Output Protocol setup
│   ├── gop.h
│   ├── memory.c            # Memory map retrieval
│   ├── memory.h
│   └── Makefile
├── common/
│   └── boot_info.h         # Shared boot info struct
├── Makefile
└── README.md
```

---

## 1.2 Kernel Core (C and x86_64 assembly, freestanding, no libc)

Language: **C17** (with minimal inline assembly for hardware interaction), compiled with `gcc -ffreestanding -nostdlib -mno-red-zone -mcmodel=kernel`.

### 1.2.1 CPU Initialization
- GDT reload in 64-bit mode (kernel code/data, user code/data, TSS)
- IDT with all 256 entries: first 32 = CPU exceptions (with ISR stubs in assembly that push error codes uniformly), 33–47 = remapped PIC IRQs (or APIC), 48 = syscall interrupt, rest = reserved
- TSS setup for ring 0 ↔ ring 3 transitions (RSP0)
- APIC initialization (Local APIC + I/O APIC) with fallback to legacy 8259 PIC
- SMP (Symmetric Multi-Processing): detect APs via ACPI MADT, boot them with a trampoline (real mode → long mode), per-CPU data structures

### 1.2.2 Memory Management
- **Physical Memory Manager**: bitmap-based page frame allocator (4 KiB pages), initialized from the E820/UEFI memory map. Functions: `pmm_alloc_frame()`, `pmm_free_frame()`, `pmm_alloc_contiguous(count)`
- **Virtual Memory Manager**: 4-level x86_64 paging (PML4 → PDPT → PD → PT). Functions: `vmm_map_page(pml4, virt, phys, flags)`, `vmm_unmap_page(pml4, virt)`, `vmm_create_address_space()`, `vmm_switch_address_space(pml4)`
- **Kernel Heap**: slab allocator for small objects (32, 64, 128, 256, 512, 1024, 2048 bytes) + buddy allocator for larger allocations. Exposed as `kmalloc(size)`, `kfree(ptr)`, `krealloc(ptr, new_size)`
- **Memory-mapped I/O helpers**: `mmio_read32(addr)`, `mmio_write32(addr, val)`, etc.

### 1.2.3 Interrupt & Exception Handling
- Common ISR/IRQ stub in assembly that saves all registers, calls a C handler, restores registers, `iretq`
- Page fault handler: check if fault is in a valid VMA → allocate + map on demand, or segfault the process
- Double fault handler on a dedicated IST stack
- Timer IRQ (PIT or HPET or APIC timer) for preemptive scheduling

### 1.2.4 Process & Thread Management
- Process structure: PID, address space (PML4), file descriptor table, working directory, signal handlers, exit code, parent PID, children list
- Thread structure: TID, owning process, kernel stack, saved register state, scheduling state (READY, RUNNING, BLOCKED, ZOMBIE)
- Scheduler: **Completely Fair Scheduler (CFS)** — red-black tree keyed on virtual runtime, O(log n) pick-next, time slice proportional to priority/nice value
- Context switch in assembly: save callee-saved registers + RSP to old thread's TCB, load from new thread's TCB, switch CR3 if process differs
- Fork: deep-copy address space (or COW — Copy-On-Write with page fault handler)
- Exec: load a new ELF into the current process's address space
- Wait/Exit: parent–child synchronization, zombie reaping

### 1.2.5 Synchronization Primitives
- Spinlock (with `cli`/`sti` for single-CPU, ticket-based for SMP)
- Mutex (blocking, with wait queue)
- Semaphore (counting)
- Read-Write Lock
- Condition Variable
- `sleep(ms)` / `wake()` via timer + scheduler integration

### 1.2.6 System Call Interface
- `syscall`/`sysret` fast path (MSR setup for `STAR`, `LSTAR`, `SFMASK`)
- At minimum, implement the following syscalls (POSIX-inspired numbering):

| # | Name | Signature |
|---|------|-----------|
| 0 | sys_read | `ssize_t read(int fd, void *buf, size_t count)` |
| 1 | sys_write | `ssize_t write(int fd, const void *buf, size_t count)` |
| 2 | sys_open | `int open(const char *path, int flags, mode_t mode)` |
| 3 | sys_close | `int close(int fd)` |
| 4 | sys_stat | `int stat(const char *path, struct stat *buf)` |
| 5 | sys_fstat | `int fstat(int fd, struct stat *buf)` |
| 6 | sys_lseek | `off_t lseek(int fd, off_t offset, int whence)` |
| 7 | sys_mmap | `void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)` |
| 8 | sys_munmap | `int munmap(void *addr, size_t len)` |
| 9 | sys_brk | `int brk(void *addr)` |
| 10 | sys_ioctl | `int ioctl(int fd, unsigned long req, ...)` |
| 11 | sys_fork | `pid_t fork(void)` |
| 12 | sys_exec | `int execve(const char *path, char *const argv[], char *const envp[])` |
| 13 | sys_exit | `void exit(int status)` |
| 14 | sys_wait | `pid_t waitpid(pid_t pid, int *status, int options)` |
| 15 | sys_getpid | `pid_t getpid(void)` |
| 16 | sys_pipe | `int pipe(int fds[2])` |
| 17 | sys_dup2 | `int dup2(int oldfd, int newfd)` |
| 18 | sys_socket | `int socket(int domain, int type, int protocol)` |
| 19 | sys_bind | `int bind(int fd, const struct sockaddr *addr, socklen_t len)` |
| 20 | sys_listen | `int listen(int fd, int backlog)` |
| 21 | sys_accept | `int accept(int fd, struct sockaddr *addr, socklen_t *len)` |
| 22 | sys_connect | `int connect(int fd, const struct sockaddr *addr, socklen_t len)` |
| 23 | sys_send | `ssize_t send(int fd, const void *buf, size_t len, int flags)` |
| 24 | sys_recv | `ssize_t recv(int fd, void *buf, size_t len, int flags)` |
| 25 | sys_mkdir | `int mkdir(const char *path, mode_t mode)` |
| 26 | sys_rmdir | `int rmdir(const char *path)` |
| 27 | sys_unlink | `int unlink(const char *path)` |
| 28 | sys_getcwd | `char *getcwd(char *buf, size_t size)` |
| 29 | sys_chdir | `int chdir(const char *path)` |
| 30 | sys_getdents | `int getdents(int fd, struct dirent *buf, size_t count)` |
| 31 | sys_clock_gettime | `int clock_gettime(clockid_t clk, struct timespec *tp)` |
| 32 | sys_nanosleep | `int nanosleep(const struct timespec *req, struct timespec *rem)` |
| 33 | sys_kill | `int kill(pid_t pid, int sig)` |
| 34 | sys_sigaction | `int sigaction(int sig, const struct sigaction *act, struct sigaction *old)` |
| 35 | sys_mprotect | `int mprotect(void *addr, size_t len, int prot)` |
| 36 | sys_futex | `int futex(int *uaddr, int op, int val, ...)` |
| 37 | sys_shmget | `int shmget(key_t key, size_t size, int flags)` |
| 38 | sys_shmat | `void *shmat(int shmid, const void *addr, int flags)` |
| 39 | sys_poll | `int poll(struct pollfd *fds, nfds_t nfds, int timeout)` |

### Deliverables:
```
kernel/
├── arch/
│   └── x86_64/
│       ├── boot.asm           # Kernel entry point (from bootloader)
│       ├── gdt.c / gdt.h
│       ├── idt.c / idt.h
│       ├── isr_stubs.asm      # ISR/IRQ assembly stubs
│       ├── tss.c / tss.h
│       ├── apic.c / apic.h
│       ├── pic.c / pic.h
│       ├── pit.c / pit.h
│       ├── hpet.c / hpet.h
│       ├── smp.c / smp.h
│       ├── trampoline.asm     # AP bootstrap
│       ├── context_switch.asm
│       ├── syscall_entry.asm
│       └── cpu.c / cpu.h      # CPUID, MSR, CR access
├── mm/
│   ├── pmm.c / pmm.h
│   ├── vmm.c / vmm.h
│   ├── heap.c / heap.h        # Slab + buddy allocator
│   └── mmap.c / mmap.h        # Process memory mapping
├── proc/
│   ├── process.c / process.h
│   ├── thread.c / thread.h
│   ├── scheduler.c / scheduler.h  # CFS
│   ├── elf_loader.c / elf_loader.h
│   └── signal.c / signal.h
├── sync/
│   ├── spinlock.c / spinlock.h
│   ├── mutex.c / mutex.h
│   ├── semaphore.c / semaphore.h
│   ├── rwlock.c / rwlock.h
│   └── condvar.c / condvar.h
├── syscall/
│   ├── syscall.c / syscall.h  # Dispatch table
│   ├── sys_file.c             # File-related syscalls
│   ├── sys_proc.c             # Process-related syscalls
│   ├── sys_mem.c              # Memory-related syscalls
│   ├── sys_net.c              # Network-related syscalls
│   └── sys_time.c             # Time-related syscalls
├── lib/
│   ├── string.c / string.h   # memcpy, memset, strlen, strcmp, etc.
│   ├── printf.c / printf.h   # Kernel printf (to serial + framebuffer)
│   ├── list.h                 # Intrusive linked list
│   ├── rbtree.c / rbtree.h   # Red-black tree (for CFS)
│   ├── bitmap.c / bitmap.h
│   └── panic.c / panic.h     # Kernel panic with stack trace
├── main.c                     # kernel_main() entry
├── linker.ld                  # Kernel linker script (higher-half)
└── Makefile
```

---

# ============================================================
# PHASE 2 — DEVICE DRIVERS & HARDWARE ABSTRACTION
# ============================================================

## 2.1 Driver Framework
- Device tree / device manager: enumerate PCI devices, match to drivers
- PCI/PCIe bus driver: configuration space access (I/O port or MMIO ECAM), BAR mapping, MSI/MSI-X interrupt setup
- A driver model with `struct driver { const char *name; int (*probe)(struct pci_device *); void (*remove)(struct pci_device *); ... }`

## 2.2 Storage Drivers
- **AHCI (SATA)**: full driver — HBA initialization, port enumeration, IDENTIFY DEVICE, read/write (DMA with PRDs), interrupt handling
- **NVMe**: controller initialization, admin queue, I/O submission/completion queues, read/write commands
- **VirtIO-Block** (for QEMU/KVM testing): virtqueue setup, buffer descriptors, read/write

## 2.3 Filesystem Layer
- **Virtual Filesystem (VFS)**: inode, dentry, superblock, file_operations, vfs_open, vfs_read, vfs_write, vfs_close, vfs_mkdir, vfs_readdir, mount/umount
- **ext2 driver**: full read/write support — superblock, block groups, inode table, directory entries, block allocation (bitmap), file read/write, creating/deleting files and directories
- **FAT32 driver**: read/write support (needed for EFI System Partition and USB drives)
- **tmpfs**: in-memory filesystem for `/tmp`
- **devfs**: virtual filesystem exposing device nodes at `/dev`
- **procfs**: virtual filesystem at `/proc` exposing process info, memory stats, CPU info

## 2.4 Input Drivers
- **PS/2 Keyboard**: IRQ1 handler, scancode set 1 → keycode translation, key event queue
- **PS/2 Mouse**: IRQ12 handler, packet parsing (3-byte + optional scroll), mouse event queue
- **USB HID**: via the USB stack (see below) — keyboard and mouse support

## 2.5 USB Stack
- **xHCI** (USB 3.x) host controller driver: register initialization, device context management, transfer rings, event rings, command rings
- USB enumeration: device descriptors, configuration descriptors, interface descriptors, endpoint descriptors
- USB mass storage (bulk-only transport) — for USB drives
- USB HID class driver — for keyboards and mice

## 2.6 Network Drivers
- **Intel E1000/E1000e** (for QEMU and real Intel NICs): initialization, TX/RX descriptor rings, DMA buffer management, interrupt handling, link status
- **VirtIO-Net** (for QEMU/KVM): virtqueue-based TX/RX
- **RTL8139** (simple, good for testing): I/O port-based, ring buffer RX, single-descriptor TX

## 2.7 Graphics Drivers
- **Framebuffer driver** (generic): use the linear framebuffer from boot (VESA/GOP), implement `put_pixel`, `fill_rect`, `blit`, `scroll`, double buffering
- **VirtIO-GPU** (for QEMU): 2D acceleration (TRANSFER_TO_HOST_2D, SET_SCANOUT, RESOURCE_FLUSH), 3D (Virgl) command submission for OpenGL acceleration
- **Basic Intel HD Graphics / AMDGPU stub**: mode setting via the display engine (read EDID, set resolution), scanout surface configuration, basic 2D blit engine — enough to set a mode and display a framebuffer; mark 3D command streamer as TODO for expansion

## 2.8 Audio Driver
- **Intel HDA (High Definition Audio)**: codec discovery, widget enumeration, output path setup, PCM playback with DMA (BDL — Buffer Descriptor List), mixer controls
- **AC97** fallback (simpler, for older VMs)

## 2.9 RTC & Timers
- CMOS RTC: read date/time
- PIT (8254): channel 0 for system timer
- HPET: as primary high-resolution timer if available
- APIC timer: per-CPU timer for scheduling

### Deliverables:
```
drivers/
├── pci/
│   ├── pci.c / pci.h
│   └── pci_ids.h
├── storage/
│   ├── ahci.c / ahci.h
│   ├── nvme.c / nvme.h
│   └── virtio_blk.c / virtio_blk.h
├── block/
│   └── block_dev.c / block_dev.h   # Block layer abstraction
├── fs/
│   ├── vfs.c / vfs.h
│   ├── ext2.c / ext2.h
│   ├── fat32.c / fat32.h
│   ├── tmpfs.c / tmpfs.h
│   ├── devfs.c / devfs.h
│   └── procfs.c / procfs.h
├── input/
│   ├── ps2_keyboard.c / ps2_keyboard.h
│   ├── ps2_mouse.c / ps2_mouse.h
│   └── input_event.h
├── usb/
│   ├── xhci.c / xhci.h
│   ├── usb_core.c / usb_core.h
│   ├── usb_hid.c / usb_hid.h
│   └── usb_storage.c / usb_storage.h
├── net/
│   ├── e1000.c / e1000.h
│   ├── virtio_net.c / virtio_net.h
│   └── rtl8139.c / rtl8139.h
├── gpu/
│   ├── framebuffer.c / framebuffer.h
│   ├── virtio_gpu.c / virtio_gpu.h
│   └── intel_gpu_stub.c / intel_gpu_stub.h
├── audio/
│   ├── hda.c / hda.h
│   └── ac97.c / ac97.h
├── timer/
│   ├── rtc.c / rtc.h
│   ├── pit.c / pit.h
│   └── hpet.c / hpet.h
└── Makefile
```

---

# ============================================================
# PHASE 3 — NETWORKING STACK
# ============================================================

Implement a full TCP/IP networking stack:

## 3.1 Link Layer
- Ethernet frame parsing and construction (src MAC, dst MAC, EtherType)
- ARP: request/reply, ARP cache with timeout

## 3.2 Network Layer
- **IPv4**: packet construction/parsing, header checksum, fragmentation/reassembly
- **ICMP**: echo request/reply (ping), destination unreachable, time exceeded
- **Routing table**: longest prefix match, default gateway, `route_add`, `route_del`, `route_lookup`
- **IPv6** (basic): packet parsing, NDP (Neighbor Discovery Protocol), stateless autoconfiguration — enough to browse IPv6 websites

## 3.3 Transport Layer
- **UDP**: send/receive, port multiplexing, checksum
- **TCP**: full implementation:
  - 3-way handshake (SYN, SYN-ACK, ACK)
  - Data transfer with sequence/acknowledgment numbers
  - Sliding window flow control
  - Congestion control (slow start, congestion avoidance, fast retransmit, fast recovery — NewReno)
  - Retransmission timer (Karn's algorithm, exponential backoff)
  - FIN/RST connection teardown
  - TIME_WAIT handling
  - Out-of-order segment reassembly
  - Nagle's algorithm (with TCP_NODELAY option)
  - Keep-alive

## 3.4 Socket API
- BSD socket interface exposed via syscalls: `socket`, `bind`, `listen`, `accept`, `connect`, `send`, `recv`, `sendto`, `recvfrom`, `close`, `setsockopt`, `getsockopt`, `select`/`poll`
- Support for `AF_INET` (IPv4), `AF_INET6` (IPv6), `AF_UNIX` (local IPC)
- `SOCK_STREAM` (TCP), `SOCK_DGRAM` (UDP), `SOCK_RAW`

## 3.5 Application-Layer Protocols (in kernel or as services)
- **DHCP client**: auto-configure IP, subnet mask, gateway, DNS servers
- **DNS resolver**: recursive/iterative queries, A/AAAA/CNAME records, response caching with TTL

### Deliverables:
```
net/
├── ethernet.c / ethernet.h
├── arp.c / arp.h
├── ipv4.c / ipv4.h
├── ipv6.c / ipv6.h
├── icmp.c / icmp.h
├── udp.c / udp.h
├── tcp.c / tcp.h
├── socket.c / socket.h
├── route.c / route.h
├── dhcp.c / dhcp.h
├── dns.c / dns.h
├── netbuf.c / netbuf.h    # Network buffer (like sk_buff)
└── Makefile
```

---

# ============================================================
# PHASE 4 — USERSPACE C LIBRARY (libc)
# ============================================================

Implement a minimal POSIX-compatible C library that userspace programs link against:

## 4.1 Core
- `_start` entry point (calls `__libc_init`, then `main`, then `exit`)
- System call wrappers (inline assembly for `syscall` instruction)
- errno handling (thread-local via `%fs` segment)

## 4.2 Standard I/O (`<stdio.h>`)
- `FILE` structure with buffering (full, line, none)
- `fopen`, `fclose`, `fread`, `fwrite`, `fseek`, `ftell`, `rewind`
- `printf`, `fprintf`, `sprintf`, `snprintf` (full format specifier support: `%d`, `%u`, `%x`, `%o`, `%s`, `%c`, `%f`, `%e`, `%g`, `%p`, `%ld`, `%lld`, `%zu`, width, precision, flags)
- `scanf`, `fscanf`, `sscanf`
- `stdin`, `stdout`, `stderr` pre-opened

## 4.3 Memory (`<stdlib.h>`, `<string.h>`)
- `malloc`, `free`, `calloc`, `realloc` — implemented via `mmap`/`brk` syscalls with a dlmalloc-style allocator
- `memcpy`, `memmove`, `memset`, `memcmp`
- `strlen`, `strcpy`, `strncpy`, `strcat`, `strncat`, `strcmp`, `strncmp`, `strchr`, `strrchr`, `strstr`, `strtok`
- `atoi`, `atol`, `strtol`, `strtoul`, `strtoll`, `strtoull`, `strtod`

## 4.4 Process (`<unistd.h>`, `<sys/wait.h>`)
- `fork`, `exec`, `execvp` (with PATH search), `waitpid`, `_exit`
- `getpid`, `getppid`, `getuid`, `getgid`
- `pipe`, `dup`, `dup2`
- `chdir`, `getcwd`
- `sleep`, `usleep`, `nanosleep`

## 4.5 Threads (`<pthread.h>`)
- `pthread_create`, `pthread_join`, `pthread_exit`, `pthread_self`
- `pthread_mutex_init/lock/unlock/destroy`
- `pthread_cond_init/wait/signal/broadcast/destroy`
- `pthread_rwlock_*`
- Thread-local storage (`__thread` / TLS via `%fs`)

## 4.6 Networking (`<sys/socket.h>`, `<netinet/in.h>`, `<arpa/inet.h>`)
- Socket API wrappers
- `inet_aton`, `inet_ntoa`, `inet_pton`, `inet_ntop`
- `getaddrinfo`, `freeaddrinfo` (using kernel DNS resolver)
- `htons`, `ntohs`, `htonl`, `ntohl`

## 4.7 Math (`<math.h>`)
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sqrt`, `cbrt`, `pow`, `exp`, `log`, `log2`, `log10`
- `floor`, `ceil`, `round`, `fabs`, `fmod`
- Use x87 FPU or SSE2 intrinsics

## 4.8 Dynamic Linking (`<dlfcn.h>`)
- ELF dynamic linker (`ld-aurora.so`): loads shared libraries, resolves symbols, performs relocations (R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT, R_X86_64_RELATIVE), lazy binding via PLT/GOT
- `dlopen`, `dlsym`, `dlclose`, `dlerror`

### Deliverables:
```
libc/
├── crt/
│   ├── crt0.asm           # _start
│   └── crti.asm / crtn.asm
├── syscall/
│   └── syscall.h          # Inline syscall wrappers
├── stdio/
│   ├── printf.c
│   ├── scanf.c
│   ├── file.c
│   └── stdio.h
├── stdlib/
│   ├── malloc.c
│   ├── atoi.c
│   ├── strtol.c
│   └── stdlib.h
├── string/
│   ├── memcpy.c / memset.c / memmove.c
│   ├── strlen.c / strcmp.c / strcpy.c / ...
│   └── string.h
├── unistd/
│   ├── fork.c / exec.c / pipe.c / ...
│   └── unistd.h
├── pthread/
│   ├── pthread.c
│   ├── mutex.c
│   ├── condvar.c
│   └── pthread.h
├── socket/
│   ├── socket.c
│   ├── inet.c
│   └── socket.h / netinet_in.h / arpa_inet.h
├── math/
│   ├── trig.c / exp.c / pow.c / ...
│   └── math.h
├── dynlink/
│   ├── ld_aurora.c         # Dynamic linker
│   └── dlfcn.c / dlfcn.h
├── include/
│   ├── errno.h
│   ├── signal.h
│   ├── fcntl.h
│   ├── sys/
│   │   ├── types.h
│   │   ├── stat.h
│   │   ├── mman.h
│   │   ├── wait.h
│   │   └── ioctl.h
│   └── ... (all standard POSIX headers)
├── Makefile
└── libc.ld                 # Shared library linker script
```

---

# ============================================================
# PHASE 5 — GRAPHICS STACK & WINDOW SYSTEM
# ============================================================

## 5.1 Display Server (AuroraCompositor — Wayland-inspired)
- Runs as a privileged userspace daemon
- Communicates with clients via Unix domain sockets + shared memory buffers
- Protocol messages (binary):
  - `AURORA_CREATE_WINDOW { width, height, title }` → `window_id`
  - `AURORA_DESTROY_WINDOW { window_id }`
  - `AURORA_ATTACH_BUFFER { window_id, shm_id, width, height, stride, format }`
  - `AURORA_COMMIT { window_id }` — display the attached buffer
  - `AURORA_MOVE_WINDOW { window_id, x, y }`
  - `AURORA_RESIZE_WINDOW { window_id, width, height }`
  - `AURORA_INPUT_EVENT { type, keycode/button, x, y, modifiers }`
  - `AURORA_DAMAGE { window_id, x, y, w, h }`
- Compositor: alpha-blended compositing of all visible windows onto the screen framebuffer
  - Window Z-ordering, focus management
  - Damage tracking (only re-composite changed regions)
  - VSync-aligned page flipping (double buffering with the GPU framebuffer)
  - Hardware cursor sprite (if supported)

## 5.2 Client Library (libaurora)
- Userspace library that wraps the compositor protocol
- `aurora_connect()` → compositor connection
- `aurora_create_window(width, height, title)` → `AuroraWindow *`
- `aurora_window_get_buffer(window)` → `void *pixels` (shared memory region)
- `aurora_window_commit(window)`
- `aurora_poll_event(event *)` — get input events
- `aurora_disconnect()`

## 5.3 Widget Toolkit (libaurora-ui)
- Built on top of `libaurora`
- Software-rendered, anti-aliased vector graphics (using a 2D rasterizer)
- Widgets: Window, Button, Label, TextInput, TextArea, Checkbox, RadioButton, Slider, ScrollBar, ListView, TabBar, MenuBar, PopupMenu, FileDialog, MessageBox, ProgressBar, Image
- Layout system: horizontal box, vertical box, grid, absolute positioning
- Theming: JSON-based theme files (colors, fonts, border radii, shadows)
- Font rendering: TrueType font parser + rasterizer (scan-line, or use stb_truetype as a reference), subpixel rendering, text shaping
- Event system: click, hover, focus, key press, scroll, drag-and-drop
- Default modern dark theme with glassmorphism-inspired translucency

## 5.4 OpenGL Implementation (software + hardware paths)
- **Software renderer (Mesa-like)**: a basic OpenGL 2.1 / OpenGL ES 2.0 compatible software rasterizer:
  - Vertex transformation (model-view-projection matrices)
  - Triangle rasterization (scanline or half-space)
  - Texture mapping (nearest, bilinear filtering)
  - Depth buffering
  - Alpha blending
  - Basic GLSL vertex/fragment shader interpreter (compile GLSL to an internal bytecode, interpret)
- **Hardware-accelerated path** (VirtIO-GPU 3D / Virgl):
  - Translate OpenGL calls to Virgl protocol commands
  - Submit to VirtIO-GPU via `VIRTIO_GPU_CMD_SUBMIT_3D`
- Exposed as `libGL.so` with standard OpenGL entry points

### Deliverables:
```
graphics/
├── compositor/
│   ├── main.c              # AuroraCompositor daemon
│   ├── compositor.c / .h   # Window management, compositing
│   ├── protocol.h          # Wire protocol definitions
│   ├── input.c / .h        # Input dispatch
│   └── cursor.c / .h       # Hardware/software cursor
├── libaurora/
│   ├── aurora.c / aurora.h # Client library
│   ├── shm.c / shm.h      # Shared memory buffer management
│   └── Makefile
├── libaurora-ui/
│   ├── widget.c / widget.h
│   ├── button.c / label.c / textinput.c / ...
│   ├── layout.c / layout.h
│   ├── theme.c / theme.h
│   ├── font.c / font.h     # TrueType rasterizer
│   ├── canvas.c / canvas.h # 2D drawing primitives
│   └── Makefile
├── opengl/
│   ├── gl.c / gl.h         # OpenGL API entry points
│   ├── rasterizer.c / .h   # Software triangle rasterizer
│   ├── shader.c / .h       # GLSL compiler + interpreter
│   ├── texture.c / .h
│   ├── matrix.c / .h       # Mat4 operations
│   ├── virgl.c / .h        # Hardware-accelerated path
│   └── Makefile
└── Makefile
```

---

# ============================================================
# PHASE 6 — CORE USERSPACE APPLICATIONS
# ============================================================

## 6.1 Init System
- PID 1 process
- Parses `/etc/aurora/init.conf` for service definitions
- Starts services in dependency order (parallel where possible)
- Manages service lifecycle (start, stop, restart, enable, disable)
- Reaps orphaned zombie processes
- Handles system shutdown/reboot

## 6.2 Shell (AuroraShell — ash)
- Interactive command-line shell with:
  - Command parsing (pipes `|`, redirections `>`, `>>`, `<`, `2>`, `&>`, background `&`)
  - Environment variables (`$VAR`, `export`, `unset`)
  - Built-in commands: `cd`, `pwd`, `echo`, `export`, `unset`, `exit`, `alias`, `unalias`, `history`, `source`, `type`, `jobs`, `fg`, `bg`, `kill`, `set`
  - Scripting: `if/elif/else/fi`, `for/do/done`, `while/do/done`, `case/esac`, functions, `$()` command substitution, `$(())` arithmetic
  - Tab completion (commands, files, directories)
  - Command history (up/down arrows, `~/.ash_history`)
  - Line editing (left/right arrows, Home, End, Delete, Backspace, Ctrl+A/E/K/U/W)
  - Signal handling (Ctrl+C → SIGINT, Ctrl+Z → SIGTSTP)
  - Prompt customization (`$PS1`)
  - Globbing (`*`, `?`, `[...]`)
  - Job control

## 6.3 Core Utilities (one binary each, or a busybox-style multicall binary)
`ls`, `cat`, `cp`, `mv`, `rm`, `mkdir`, `rmdir`, `touch`, `chmod`, `chown`, `ln`, `find`, `grep`, `sed`, `awk`, `sort`, `uniq`, `wc`, `head`, `tail`, `tee`, `xargs`, `tr`, `cut`, `paste`, `diff`, `tar`, `gzip`/`gunzip`, `mount`, `umount`, `df`, `du`, `ps`, `top`, `kill`, `uname`, `whoami`, `hostname`, `date`, `cal`, `clear`, `env`, `true`, `false`, `yes`, `sleep`, `test`/`[`, `expr`, `seq`, `basename`, `dirname`, `realpath`, `readlink`, `stat`, `file`, `hexdump`, `strings`, `id`, `su`, `passwd`, `login`, `init`, `shutdown`, `reboot`, `dmesg`, `free`, `uptime`, `lspci`, `lsusb`, `ifconfig`, `ping`, `traceroute`, `nc` (netcat), `curl` (basic HTTP GET/POST), `ssh` (basic client), `scp`

## 6.4 Terminal Emulator (AuroraTerminal)
- Graphical terminal emulator using `libaurora-ui`
- VT100/xterm escape sequence support (cursor movement, colors — 16, 256, and 24-bit, bold, underline, blink, inverse, window title)
- Scrollback buffer (configurable, default 10,000 lines)
- Font: bundled monospace TrueType font (e.g., a subset of a libre monospace font)
- Copy/paste (Ctrl+Shift+C / Ctrl+Shift+V)
- Tabs (Ctrl+Shift+T new tab, Ctrl+Shift+W close tab)
- Configurable (font size, color scheme, transparency)
- PTY (pseudo-terminal) support in the kernel (`/dev/ptmx`, `/dev/pts/*`)

## 6.5 File Manager (AuroraFiles)
- Graphical dual-pane file manager
- Icon view and list view
- File operations: copy, move, delete (to trash), rename, create file/folder
- File previews (text, images)
- Breadcrumb navigation bar
- Search
- Bookmarks sidebar (Home, Desktop, Documents, Downloads, Trash)
- Mount point display
- Context menu (right-click)
- Properties dialog (permissions, size, type)

## 6.6 Text Editor (AuroraEdit)
- Graphical text editor with:
  - Syntax highlighting (C, Python, JavaScript, Rust, Markdown, JSON, XML, shell scripts)
  - Line numbers
  - Find & replace (with regex)
  - Multiple tabs
  - Undo/redo (unlimited)
  - Auto-indent
  - Word wrap toggle
  - Status bar (line:col, encoding, language)
  - File tree sidebar
  - Minimap
  - Configurable font and theme

## 6.7 System Monitor (AuroraMonitor)
- Real-time graphs: CPU usage (per-core), memory usage, network throughput, disk I/O
- Process list with columns: PID, Name, CPU%, MEM%, State, Priority
- Sort by any column
- Kill process button
- System info panel (OS version, kernel version, CPU model, RAM total, uptime)

## 6.8 Settings Application (AuroraSettings)
- Display: resolution, refresh rate, scaling, wallpaper
- Network: Wi-Fi (if applicable), Ethernet (DHCP/static IP config)
- Sound: output device, volume, input device
- Appearance: theme (dark/light), accent color, font
- Date & Time: timezone, NTP toggle
- Users: add/remove users, change password
- About: system info

### Deliverables:
```
userspace/
├── init/
│   ├── init.c
│   └── init.conf           # Default init configuration
├── shell/
│   ├── ash.c               # Main shell
│   ├── parser.c / parser.h
│   ├── builtin.c / builtin.h
│   ├── job.c / job.h
│   ├── history.c / history.h
│   └── completion.c / completion.h
├── coreutils/
│   ├── ls.c / cat.c / cp.c / mv.c / rm.c / ... (one per utility)
│   └── Makefile
├── terminal/
│   ├── main.c
│   ├── vt100.c / vt100.h  # Escape sequence parser
│   ├── pty.c / pty.h      # PTY handling
│   └── Makefile
├── filemanager/
│   ├── main.c
│   └── Makefile
├── editor/
│   ├── main.c
│   ├── syntax.c / syntax.h
│   ├── buffer.c / buffer.h
│   └── Makefile
├── monitor/
│   ├── main.c
│   └── Makefile
├── settings/
│   ├── main.c
│   └── Makefile
└── Makefile
```

---

# ============================================================
# PHASE 7 — PACKAGE MANAGER (aurora-pkg)
# ============================================================

## 7.1 Package Format
- `.apkg` files = tar.gz archives containing:
  - `METADATA` — name, version, description, architecture, dependencies, conflicts, provides, size, maintainer, license
  - `FILES` — list of all installed files with checksums (SHA256)
  - `INSTALL` — pre-install/post-install/pre-remove/post-remove shell scripts
  - `data/` — the actual filesystem tree (rooted at `/`)

## 7.2 Package Manager CLI (`aurora-pkg`)
- `aurora-pkg install <package>` — resolve dependencies (topological sort), download from repository, verify signature (Ed25519), extract, run install scripts
- `aurora-pkg remove <package>` — check reverse dependencies, run remove scripts, delete files
- `aurora-pkg update` — refresh package database from repository
- `aurora-pkg upgrade` — upgrade all installed packages to latest
- `aurora-pkg search <query>` — search package database
- `aurora-pkg info <package>` — show package details
- `aurora-pkg list` — list installed packages
- `aurora-pkg build <directory>` — build a package from a build recipe

## 7.3 Repository
- HTTP-based repository server
- Repository index: compressed JSON or binary format with all package metadata
- Signed with Ed25519 (repository key)

### Deliverables:
```
pkg/
├── aurora-pkg.c             # Main CLI
├── database.c / database.h  # Local package database (/var/lib/aurora-pkg/)
├── resolver.c / resolver.h  # Dependency resolver
├── download.c / download.h  # HTTP download (using socket API)
├── archive.c / archive.h   # tar.gz extraction
├── verify.c / verify.h     # Ed25519 signature verification
├── recipe.c / recipe.h     # Build recipe parser
└── Makefile
```

---

# ============================================================
# PHASE 8 — WEB BROWSER (AuroraBrowser)
# ============================================================

## 8.1 Networking
- HTTP/1.1 client: request construction, response parsing (chunked transfer encoding, Content-Length, redirects, cookies)
- HTTPS: TLS 1.2/1.3 implementation (or port of BearSSL/mbedTLS — specify that we use a minimal TLS library)
  - TLS record layer, handshake, key exchange (ECDHE), cipher suites (AES-128-GCM, ChaCha20-Poly1305)
  - X.509 certificate parsing and chain validation (bundle a root CA store)
- HTTP/2 multiplexing (stretch goal, can stub)

## 8.2 HTML Parser
- Tokenizer (HTML5 spec-compliant state machine)
- Tree builder: construct a DOM tree (Document, Element, Text, Comment nodes)
- Handle: `<html>`, `<head>`, `<body>`, `<div>`, `<span>`, `<p>`, `<a>`, `<img>`, `<table>`, `<tr>`, `<td>`, `<th>`, `<ul>`, `<ol>`, `<li>`, `<form>`, `<input>`, `<button>`, `<textarea>`, `<select>`, `<option>`, `<h1>`–`<h6>`, `<br>`, `<hr>`, `<pre>`, `<code>`, `<blockquote>`, `<strong>`, `<em>`, `<script>`, `<style>`, `<link>`, `<meta>`, `<title>`, `<canvas>`, `<video>`, `<audio>`, `<iframe>`, `<nav>`, `<header>`, `<footer>`, `<section>`, `<article>`, `<aside>`, `<main>`
- Entity decoding (`&amp;`, `&lt;`, `&gt;`, `&#...;`, `&#x...;`)

## 8.3 CSS Engine
- CSS parser: selectors (type, class, ID, attribute, pseudo-class `:hover`/`:focus`/`:first-child`/`:nth-child`, pseudo-element `::before`/`::after`), properties, values, `@media` queries, `@import`, `@font-face`
- Cascade & specificity: user-agent stylesheet → author stylesheet → inline styles, `!important`
- Selector matching against DOM
- Computed style resolution (inheritance, initial values, `em`/`rem`/`%`/`px`/`vw`/`vh` units)
- Supported properties: `display`, `position`, `top`/`right`/`bottom`/`left`, `width`/`height`/`min-*`/`max-*`, `margin`, `padding`, `border`, `background` (color, image, gradient), `color`, `font-family`/`font-size`/`font-weight`/`font-style`, `text-align`/`text-decoration`/`text-transform`/`line-height`/`letter-spacing`, `overflow`, `z-index`, `opacity`, `visibility`, `float`/`clear`, `flexbox` (display:flex, flex-direction, justify-content, align-items, flex-grow/shrink/basis, flex-wrap), `grid` (basic), `transform` (translate, rotate, scale), `transition`, `box-shadow`, `border-radius`, `cursor`, `list-style`, `white-space`, `word-wrap`

## 8.4 Layout Engine
- **Block layout**: block-level boxes stacked vertically, margin collapsing
- **Inline layout**: inline boxes on lines, line breaking, text wrapping
- **Flexbox layout**: full flexbox spec
- **Table layout**: basic table rendering
- **Positioned layout**: relative, absolute, fixed, sticky
- **Float layout**: left/right floats with clearance
- Layout tree → display list (paint commands: fill rect, draw text, draw image, draw border, clip, transform)

## 8.5 Rendering / Painting
- Walk the display list, rasterize to a pixel buffer:
  - Anti-aliased text rendering (via font rasterizer)
  - Image decoding: PNG (deflate), JPEG (baseline DCT), GIF (LZW), WebP (basic)
  - Rounded corners (border-radius)
  - Box shadows
  - Linear/radial gradients
  - Opacity / alpha compositing
  - CSS transforms (2D matrix transform on composited layers)
  - Scrolling (main page and overflow:scroll elements)

## 8.6 JavaScript Engine (AuroraScript)
- **Lexer**: tokenize JS source (identifiers, keywords, numbers, strings, template literals, regex literals, operators, punctuation)
- **Parser**: recursive descent, produces AST (ES2020 subset):
  - Variables: `var`, `let`, `const`
  - Functions: declarations, expressions, arrow functions, default params, rest params
  - Control: `if/else`, `for`, `for...in`, `for...of`, `while`, `do...while`, `switch/case`, `try/catch/finally`, `throw`
  - Operators: arithmetic, comparison, logical, bitwise, assignment, ternary, `typeof`, `instanceof`, `in`, `delete`, `void`, spread `...`, optional chaining `?.`, nullish coalescing `??`
  - Objects: literals, property access, computed properties, shorthand, destructuring
  - Arrays: literals, destructuring, spread
  - Classes: `class`, `extends`, `constructor`, methods, static methods, getters/setters
  - Modules: `import`/`export` (basic)
  - Async: `Promise`, `async`/`await` (basic event-loop based)
  - Iterators and generators (basic)
  - Template literals
  - Regex (basic)
- **Bytecode compiler**: AST → stack-based bytecode
- **Virtual machine**: bytecode interpreter (stack-based), with:
  - Dynamic typing (Number, String, Boolean, Null, Undefined, Object, Array, Function, Symbol)
  - Prototype-based inheritance
  - Closures and scope chains
  - Garbage collector (mark-and-sweep, or mark-compact)
  - `this` binding rules (default, implicit, explicit `call`/`apply`/`bind`, `new`)

- **Built-in objects**: `Object`, `Array`, `String`, `Number`, `Boolean`, `Math`, `Date`, `RegExp`, `JSON`, `Map`, `Set`, `WeakMap`, `WeakSet`, `Promise`, `Error` (and subtypes), `Symbol`, `console`, `ArrayBuffer`, `DataView`, `TypedArrays`

- **Web APIs** (exposed to page scripts):
  - `document` (DOM API): `getElementById`, `querySelector`, `querySelectorAll`, `createElement`, `createTextNode`, `appendChild`, `removeChild`, `insertBefore`, `replaceChild`, `setAttribute`, `getAttribute`, `removeAttribute`, `classList`, `style`, `innerHTML`, `innerText`, `textContent`, `parentNode`, `children`, `nextSibling`, `previousSibling`, `addEventListener`, `removeEventListener`, `dispatchEvent`
  - `window`: `setTimeout`, `setInterval`, `clearTimeout`, `clearInterval`, `requestAnimationFrame`, `alert`, `confirm`, `prompt`, `location`, `history`, `navigator`, `screen`, `innerWidth`/`innerHeight`, `scrollTo`, `scrollBy`, `getComputedStyle`, `localStorage`, `sessionStorage`
  - `XMLHttpRequest` and `fetch` API (basic)
  - `Canvas2D` API (drawing to `<canvas>` element)
  - `Event` objects (MouseEvent, KeyboardEvent, InputEvent, FocusEvent, etc.)
  - `FormData`, `URL`, `URLSearchParams`

## 8.7 Browser UI
- Tab bar (open, close, reorder tabs)
- Address bar with URL entry
- Back / Forward / Reload / Home buttons
- Bookmarks bar
- Download manager
- Settings page (home page, default search engine, clear cookies/cache)
- View source
- Developer tools panel (basic): DOM inspector, console, network tab
- Find in page (Ctrl+F)
- Zoom (Ctrl+/-, Ctrl+0)
- Full-screen mode (F11)
- Print-to-PDF

### Deliverables:
```
browser/
├── net/
│   ├── http.c / http.h
│   ├── tls.c / tls.h       # TLS 1.2/1.3 (or thin wrapper around BearSSL)
│   ├── url.c / url.h
│   └── cookie.c / cookie.h
├── html/
│   ├── tokenizer.c / .h
│   ├── parser.c / .h
│   └── dom.c / dom.h
├── css/
│   ├── tokenizer.c / .h
│   ├── parser.c / .h
│   ├── selector.c / .h
│   ├── cascade.c / .h
│   └── properties.h
├── layout/
│   ├── box.c / box.h
│   ├── block.c / block.h
│   ├── inline.c / inline.h
│   ├── flex.c / flex.h
│   ├── table.c / table.h
│   ├── paint.c / paint.h   # Display list
│   └── layout.c / layout.h
├── render/
│   ├── rasterizer.c / .h   # Pixel buffer rendering
│   ├── text.c / text.h     # Text rendering
│   ├── image.c / image.h   # PNG/JPEG/GIF/WebP decoders
│   └── compositor.c / .h   # Layer compositing
├── js/
│   ├── lexer.c / lexer.h
│   ├── parser.c / parser.h
│   ├── ast.h
│   ├── compiler.c / .h     # AST → bytecode
│   ├── vm.c / vm.h         # Bytecode interpreter
│   ├── gc.c / gc.h         # Garbage collector
│   ├── value.c / value.h   # JS value representation
│   ├── object.c / object.h
│   ├── builtin/
│   │   ├── array.c / math_obj.c / string_obj.c / date.c / json.c / ...
│   │   └── console.c
│   └── webapi/
│       ├── dom_api.c / .h
│       ├── window_api.c / .h
│       ├── fetch_api.c / .h
│       ├── canvas2d.c / .h
│       ├── storage.c / .h
│       └── event.c / .h
├── ui/
│   ├── browser_main.c
│   ├── tab_bar.c / .h
│   ├── address_bar.c / .h
│   ├── toolbar.c / .h
│   ├── devtools.c / .h
│   └── settings.c / .h
└── Makefile
```

---

# ============================================================
# PHASE 9 — GAME SUPPORT & MULTIMEDIA
# ============================================================

## 9.1 AuroraGL Game Library (SDL-like)
A game development library providing:
- Window creation and management (via `libaurora`)
- OpenGL context creation
- 2D sprite rendering (accelerated blit, rotation, scaling)
- Input handling (keyboard, mouse, gamepad — via evdev-like interface)
- Audio mixing: load WAV/OGG files, multiple simultaneous channels, volume control, panning, basic effects (reverb stub)
- Timer / frame-rate management
- Image loading (PNG, BMP, TGA)
- TTF font rendering for games (using the same TrueType rasterizer)
- Math library: vec2/vec3/vec4, mat3/mat4, quaternions

## 9.2 Bundled Games
1. **AuroraDoom** — a DOOM-style raycasting FPS:
   - Raycasting renderer (textured walls, floor/ceiling, sprites)
   - Collision detection
   - Enemy AI (state machine: idle, chase, attack)
   - Weapon system (pistol, shotgun, plasma rifle)
   - Level format (grid-based map definition)
   - Sound effects

2. **AuroraCraft** — a basic Minecraft-style voxel game:
   - Chunk-based voxel world (16x16x256 chunks)
   - Block types (grass, dirt, stone, wood, leaves, water, sand)
   - First-person camera with WASD + mouse look
   - Block placement and destruction
   - Basic terrain generation (Perlin noise heightmap)
   - Day/night cycle (skybox color lerp)
   - Simple OpenGL rendering (chunk meshing, greedy meshing optimization, frustum culling)

3. **AuroraKart** — a simple 3D racing game:
   - Track defined as a spline with road mesh generation
   - Kart physics (acceleration, braking, steering, drift)
   - AI opponents (follow spline waypoints)
   - Power-ups (speed boost, shell)
   - Split-screen multiplayer (2-player, two viewports)

4. **Classic games**: Tetris, Snake, Minesweeper, Solitaire, Chess (with basic AI)

## 9.3 Audio Subsystem (userspace)
- Audio server daemon (`aurora-audio`):
  - Mixes audio from multiple clients
  - Resampling (linear interpolation, or sinc for quality)
  - Per-client volume control
  - Outputs to the HDA/AC97 kernel driver via `/dev/dsp` or custom audio device node
- Client library (`libaurora-audio`):
  - `audio_connect()`, `audio_play(buffer, format, rate)`, `audio_set_volume(level)`

## 9.4 Media Player (AuroraMedia)
- Audio playback: WAV, OGG Vorbis (decoder included), MP3 (decoder included or stubbed)
- Video playback: basic AVI container, Motion JPEG or Theora decoder (stretch)
- Playlist support
- Graphical UI with play/pause/stop/seek controls, volume slider, album art display

### Deliverables:
```
games/
├── libauroragl/
│   ├── window.c / input.c / audio.c / timer.c / image.c / font.c / math.c
│   ├── auroragl.h
│   └── Makefile
├── doom/
│   ├── main.c / raycaster.c / enemy.c / weapon.c / map.c / sound.c
│   └── Makefile
├── voxel/
│   ├── main.c / world.c / chunk.c / terrain.c / player.c / render.c
│   └── Makefile
├── kart/
│   ├── main.c / track.c / kart.c / physics.c / ai.c / render.c
│   └── Makefile
├── classics/
│   ├── tetris.c / snake.c / minesweeper.c / solitaire.c / chess.c
│   └── Makefile
├── audio_server/
│   ├── aurora_audio.c
│   ├── mixer.c / mixer.h
│   └── Makefile
├── libaurora_audio/
│   ├── client.c / client.h
│   └── Makefile
├── media_player/
│   ├── main.c / decoder_wav.c / decoder_ogg.c / playlist.c
│   └── Makefile
└── Makefile
```

---

# ============================================================
# PHASE 10 — OLLAMA / LOCAL AI INTEGRATION
# ============================================================

## 10.1 Ollama Runtime
Port or reimplement the core of Ollama to run natively on AuroraOS:

- **GGML/GGUF model loader**: parse GGUF file format, load quantized model weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization formats)
- **Tensor operations**: matrix multiplication (optimized with SSE/AVX/AVX2 intrinsics), element-wise operations, softmax, layer normalization, RoPE (Rotary Position Embeddings)
- **Transformer inference engine**:
  - Attention mechanism (multi-head self-attention with KV cache)
  - Feed-forward network (SwiGLU activation)
  - Token embedding and positional encoding
  - Sampling: temperature, top-k, top-p (nucleus), repetition penalty
  - Tokenizer: BPE (Byte Pair Encoding) with SentencePiece model loading
- **Model architectures**: LLaMA/LLaMA 2/LLaMA 3, Mistral, Phi, Gemma (configurable via GGUF metadata)
- **HTTP API server** (matching Ollama's API):
  - `POST /api/generate` — text generation
  - `POST /api/chat` — chat completion
  - `GET /api/tags` — list local models
  - `POST /api/pull` — download model from registry
  - `DELETE /api/delete` — remove model
  - JSON request/response format
  - Streaming responses (chunked transfer encoding)
- **CLI**: `ollama run <model>`, `ollama list`, `ollama pull <model>`, `ollama serve`
- **Memory-mapped model loading**: use `mmap` for efficient model loading without consuming excess RAM
- **Multi-threaded inference**: use pthreads for parallel matrix operations across CPU cores

## 10.2 AI Assistant App (AuroraAI)
- Graphical chat interface (using `libaurora-ui`)
- Chat history with conversation threads
- Model selector dropdown
- Streaming response display (token-by-token)
- System prompt configuration
- Code syntax highlighting in responses
- Copy response button
- Export conversation as text/markdown

### Deliverables:
```
ai/
├── ollama/
│   ├── main.c              # CLI entry point
│   ├── server.c / server.h # HTTP API server
│   ├── gguf.c / gguf.h     # GGUF model file parser
│   ├── tensor.c / tensor.h # Tensor operations
│   ├── simd.c / simd.h     # SSE/AVX optimized kernels
│   ├── transformer.c / .h  # Transformer forward pass
│   ├── attention.c / .h    # Multi-head attention with KV cache
│   ├── sampling.c / .h     # Token sampling strategies
│   ├── tokenizer.c / .h    # BPE tokenizer
│   ├── model.c / model.h   # Model management (load, unload, list)
│   └── Makefile
├── aurora_ai/
│   ├── main.c              # Chat GUI
│   ├── chat.c / chat.h     # Chat logic, history
│   └── Makefile
└── Makefile
```

---

# ============================================================
# PHASE 11 — SECURITY & USER MANAGEMENT
# ============================================================

## 11.1 User System
- `/etc/passwd`: username, UID, GID, home directory, shell
- `/etc/shadow`: password hashes (SHA-512 with salt)
- `/etc/group`: group definitions
- Login process: `getty` → `login` → authenticate → spawn user shell
- `su` / `sudo` (basic: check group membership, prompt for password)

## 11.2 File Permissions
- Unix permission model: owner/group/other, rwx bits
- `chmod`, `chown`, `chgrp` enforcement in VFS
- Sticky bit, setuid, setgid

## 11.3 Process Isolation
- Separate address spaces (ring 3 for userspace, ring 0 for kernel)
- SMEP (Supervisor Mode Execution Prevention) enabled
- SMAP (Supervisor Mode Access Prevention) enabled
- NX bit enforced on data pages
- ASLR (Address Space Layout Randomization): randomize stack, heap, mmap base, executable base for PIE binaries
- Stack canaries (in libc)

## 11.4 Secure Boot Chain (optional, advanced)
- Verify kernel signature before boot (Ed25519)
- Kernel module signature verification

---

# ============================================================
# PHASE 12 — BUILD SYSTEM & DISK IMAGE CREATION
# ============================================================

## 12.1 Build System
- Top-level `Makefile` that orchestrates the entire build:
  - Cross-compiler toolchain setup (using an `x86_64-auroraos` GCC cross-compiler, or instructions to build one with crosstool-ng)
  - Bootloader build
  - Kernel build
  - libc build
  - All userspace application builds
  - Produces individual ELF binaries

## 12.2 Disk Image Creation (`make image`)
- Create a GPT-partitioned disk image:
  - Partition 1: ESP (EFI System Partition, FAT32, 100 MiB) — contains UEFI bootloader
  - Partition 2: Root filesystem (ext2, rest of disk) — contains kernel, libc, all userspace binaries, config files
- Install GRUB or use the custom bootloader
- Filesystem layout:
```
/
├── boot/
│   └── aurora-kernel       # Kernel ELF
├── bin/                    # Core utilities
├── sbin/                   # System binaries (init, mount, etc.)
├── lib/
│   ├── libc.so
│   ├── libaurora.so
│   ├── libaurora-ui.so
│   ├── libGL.so
│   ├── libaurora-audio.so
│   └── ld-aurora.so        # Dynamic linker
├── etc/
│   ├── aurora/
│   │   └── init.conf
│   ├── passwd
│   ├── shadow
│   ├── group
│   ├── hostname
│   ├── resolv.conf
│   └── fstab
├── usr/
│   ├── bin/                # User applications (browser, editor, etc.)
│   ├── lib/                # Additional libraries
│   ├── share/
│   │   ├── fonts/          # TrueType fonts
│   │   ├── themes/         # UI themes
│   │   ├── wallpapers/     # Desktop wallpapers
│   │   └── icons/          # Application icons
│   └── include/            # Development headers
├── home/
│   └── user/               # Default user home
│       ├── Desktop/
│       ├── Documents/
│       ├── Downloads/
│       └── .ash_history
├── var/
│   ├── log/                # System logs
│   └── lib/
│       └── aurora-pkg/     # Package database
├── tmp/                    # tmpfs mount point
├── dev/                    # devfs mount point
├── proc/                   # procfs mount point
└── mnt/                    # Mount points for removable media
```

## 12.3 Testing with QEMU
Provide exact commands to test:
```bash
# Basic boot test (UEFI):
qemu-system-x86_64 -bios /usr/share/OVMF/OVMF_CODE.fd \
  -drive file=auroraos.img,format=raw \
  -m 4G -smp 4 \
  -device virtio-gpu-pci \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::8080-:8080 \
  -device virtio-blk-pci,drive=hd0 \
  -drive id=hd0,file=auroraos.img,format=raw,if=none \
  -device intel-hda -device hda-duplex \
  -usb -device usb-tablet \
  -serial stdio

# With KVM acceleration:
# Add -enable-kvm -cpu host
```

### Deliverables:
```
build/
├── Makefile                 # Top-level build orchestration
├── toolchain/
│   └── build_toolchain.sh   # Cross-compiler build script
├── image/
│   ├── create_image.sh      # Disk image creation script
│   └── install_files.sh     # File installation script
├── config/
│   ├── kernel.config        # Kernel build configuration
│   └── default_theme.json   # Default UI theme
└── README.md                # Complete build instructions
```

---

# ============================================================
# PHASE 13 — DESKTOP ENVIRONMENT & POLISH
# ============================================================

## 13.1 Desktop Shell (AuroraDesktop)
- **Wallpaper**: rendered behind all windows (loads PNG/JPEG from `/usr/share/wallpapers/`)
- **Taskbar** (bottom of screen):
  - Application menu button (left) — categorized app launcher
  - Running application buttons (center) — click to focus/minimize
  - System tray (right) — clock, network status icon, volume icon, battery icon (if applicable), notification area
- **Window decorations** (drawn by compositor):
  - Title bar with app icon, title text, minimize/maximize/close buttons
  - Window dragging by title bar
  - Window resizing by edges/corners
  - Double-click title bar to maximize/restore
  - Window snapping (drag to edge: left half, right half, maximize)
- **Desktop icons**: clickable icons on the wallpaper for common locations
- **Right-click desktop menu**: New Folder, New File, Paste, Display Settings, Terminal
- **Alt+Tab** application switcher with window previews
- **Virtual workspaces** (4 workspaces, switchable via Ctrl+Alt+Arrow or a workspace indicator in taskbar)
- **Notifications**: toast-style notifications in top-right corner
- **Lock screen**: triggered by timeout or Super+L, password entry to unlock

## 13.2 Login Screen (AuroraLogin)
- Graphical login screen (runs before desktop shell)
- User selection (with avatar)
- Password entry
- Session type selector (if multiple desktop sessions available)
- Shutdown / Reboot buttons

## 13.3 Default Theme & Assets
- Dark theme with rounded corners, subtle shadows, translucent panels
- Accent color: electric blue (#0066FF) with gradients
- Bundled font: a clean sans-serif (include a libre font like Inter or Noto Sans subset)
- Bundled monospace font: for terminal and code editor
- Application icons: simple, flat-style SVG-inspired (rasterized to multiple sizes)
- Default wallpaper: a beautiful abstract/space-themed image (generate procedurally — gradient with stars and nebula)

---

# ============================================================
# GLOBAL CONSTRAINTS & CODING STANDARDS
# ============================================================

1. **Language**: C17 for all kernel and most userspace code. x86_64 NASM assembly where hardware interaction requires it. No C++ unless absolutely necessary for a specific library port.

2. **No external dependencies in kernel**: The kernel must be fully freestanding. No glibc, no POSIX, no third-party libraries. Everything from scratch.

3. **Userspace may bundle minimal third-party code** (with attribution):
   - `stb_truetype.h` for TrueType font parsing (or implement from scratch)
   - `stb_image.h` for image loading (or implement from scratch)
   - BearSSL or mbedTLS for TLS (or implement core TLS from scratch)
   - Preferably implement everything from scratch for educational completeness.

4. **Code quality**:
   - Every function must have a doc comment explaining purpose, parameters, and return value
   - Consistent naming: `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for macros and constants, `PascalCase` for type names (structs, enums)
   - Error handling: every function that can fail returns an error code or NULL; callers must check
   - No memory leaks: every `kmalloc`/`malloc` has a corresponding `kfree`/`free` path
   - Kernel code must be preemption-safe: hold spinlocks for minimum duration, disable interrupts only when necessary

5. **Build**: Must compile with `gcc` (or a cross-compiler) and `nasm`. Build with `make`. The entire OS (bootloader + kernel + libc + all userspace apps) must build from a single `make` invocation at the project root.

6. **Testing**: Include a basic kernel unit test framework (assert macros, test runner) and userspace test programs.

7. **Documentation**: Each major subsystem directory must contain a `README.md` explaining the architecture, key data structures, and how to extend it.

---

# ============================================================
# OUTPUT FORMAT
# ============================================================

For each phase, produce:

1. **All source files** with complete, compilable code (not pseudocode). Use clearly marked `// TODO: expand — [description of what needs to be implemented]` for sections that would exceed output limits, but ensure the code compiles and runs in a degraded mode without those sections.

2. **Makefile** for that phase's component.

3. **Brief architecture notes** (as comments in the code and/or a README) explaining key design decisions.

Start with **Phase 1** and produce the complete bootloader and kernel core. Then proceed phase by phase. If I say "continue", produce the next phase. If I say "expand [component]", fill in the TODO stubs for that component.

**BEGIN. Produce Phase 1 now — the complete UEFI/BIOS bootloader and kernel core for AuroraOS.**

---

## END OF PROMPT
