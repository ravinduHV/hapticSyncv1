#pragma once
// shm.h — Shared memory helpers (POSIX shm on Linux, malloc fallback elsewhere)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <cstdlib>
#endif

namespace hs {

/// Create a new shared memory segment. Returns mapped pointer or nullptr.
inline void* shm_create(const char* name, size_t size) {
#ifdef __linux__
    int fd = ::shm_open(name, O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) { perror("shm_open create"); return nullptr; }
    if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
        perror("ftruncate"); close(fd); return nullptr;
    }
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return nullptr; }
    std::memset(p, 0, size);
    return p;
#else
    (void)name;
    void* p = std::calloc(1, size);
    return p;
#endif
}

/// Open an existing shared memory segment read-only. Returns mapped pointer or nullptr.
inline void* shm_open_readonly(const char* name, size_t size) {
#ifdef __linux__
    int fd = ::shm_open(name, O_RDONLY, 0);
    if (fd < 0) { perror("shm_open readonly"); return nullptr; }
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap ro"); return nullptr; }
    return p;
#else
    (void)name; (void)size;
    return nullptr; // no shm in non-Linux fallback
#endif
}

/// Open an existing shared memory segment read-write.
inline void* shm_open_readwrite(const char* name, size_t size) {
#ifdef __linux__
    int fd = ::shm_open(name, O_RDWR, 0);
    if (fd < 0) { perror("shm_open rw"); return nullptr; }
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap rw"); return nullptr; }
    return p;
#else
    (void)name; (void)size;
    return nullptr;
#endif
}

/// Unmap and unlink a shared memory segment.
inline void shm_destroy(const char* name, void* ptr, size_t size) {
#ifdef __linux__
    if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
    if (name) ::shm_unlink(name);
#else
    (void)name; (void)size;
    std::free(ptr);
#endif
}

/// Unmap only (don't unlink — for readers).
inline void shm_close(void* ptr, size_t size) {
#ifdef __linux__
    if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
#else
    (void)size;
    // Don't free — we didn't allocate in read-only mode
#endif
}

} // namespace hs

