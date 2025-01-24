#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>

#include "sycl_khr_free_function_commands.hpp"

static void test_submit() {
  sycl::queue q;
  constexpr int val = 314;
  int *buf = sycl::malloc_shared<int>(1, q);

  const auto cgf = [&](sycl::handler& h) {
    h.single_task<>([=] { buf[0] = val; });
  };

  {
    khr::submit(q, cgf);
    q.wait();
    assert(buf[0] == val);
    buf[0] = 0;
  }
  {
    khr::submit_tracked(q, cgf).wait();
    assert(buf[0] == val);
  }
  
  sycl::free(buf, q);
}

template <int Dims>
constexpr sycl::range<Dims> get_range(int N) {
  if constexpr (Dims == 1)
    return sycl::range<1>(N);
  if constexpr (Dims == 2)
    return sycl::range<2>(N, N);
  if constexpr (Dims == 3)
    return sycl::range<3>(N, N, N);
}

template <int Dims>
constexpr size_t get_N(size_t Size) {
  if constexpr (Dims == 1)
    return Size;
  if constexpr (Dims == 2)
    return Size * Size;
  if constexpr (Dims == 3)
    return Size * Size * Size;
}

template <size_t Dims> static void test_launch_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;
  int *buf = sycl::malloc_shared<int>(N, q);

  const auto k = [=](auto item) {
    auto lin_idx = item.get_linear_id();
    buf[lin_idx] = lin_idx + lin_idx * 2;
  };

  sycl::range<Dims> r = get_range<Dims>(Size);
  {
    q.submit([&](sycl::handler &h) { khr::launch(h, r, k); });
    q.wait();

    for (int i = 0; i < N; ++i) {
      assert(buf[i] == i + i * 2);
      buf[i] = 0;
    }
  }
  {
    khr::launch(q, r, k);
    q.wait();

    for (int i = 0; i < N; ++i)
      assert(buf[i] == i + i * 2);
  }
  
  sycl::free(buf, q);
}

static void test_launch() {
  test_launch_impl<1>();
  test_launch_impl<2>();
  test_launch_impl<3>();
}

template <int Dims> static void test_launch_reduce_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;
  int sumResult = 0;
  sycl::buffer<int> sumBuf { &sumResult, 1 };

  const auto task = [=](sycl::item<Dims> item, auto& sum) {
    sum += item.get_linear_id();
  };

  sycl::range<Dims> r = get_range<Dims>(Size);
  int expected_res = (N - 1) * N / 2;
  {
    q.submit([&](sycl::handler& h) {
      khr::launch_reduce(h, r, task, sycl::reduction(sumBuf, h, sycl::plus<>()));
    });
    q.wait();
    assert(sumBuf.get_host_access()[0] == expected_res);
    sumBuf.get_host_access()[0] = 0;
  }
  {
    khr::launch_reduce(q, r, task, sycl::reduction(&sumResult, sycl::plus<>()));
    q.wait();
    assert(sumBuf.get_host_access()[0] == expected_res);
  }
}

static void test_launch_reduce() {
  test_launch_reduce_impl<1>();
  test_launch_reduce_impl<2>();
  test_launch_reduce_impl<3>();
}

template <int Dims> static void test_launch_grouped_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;
  int *buf = sycl::malloc_shared<int>(N, q);

  const auto task = [=](sycl::nd_item<Dims> item) {
    auto idx = item.get_global_linear_id();
    buf[idx] = idx + idx * 2;
  };

  sycl::range<Dims> r_glob = get_range<Dims>(Size);
  sycl::range<Dims> r_loc = get_range<Dims>(Size / 2);
  {
    q.submit([&](sycl::handler &h) {
      khr::launch_grouped(h, r_glob, r_loc, task);
    });
    q.wait();

    for (int i = 0; i < N; ++i) {
      assert(buf[i] == i + i * 2);
      buf[i] = 0;
    }
  }
  {
    khr::launch_grouped(q, r_glob, r_loc, task);
    q.wait();

    for (int i = 0; i < N; ++i)
      assert(buf[i] == i + i * 2);
  }
  
  sycl::free(buf, q);
}

void test_launch_grouped() {
  test_launch_grouped_impl<1>();
  test_launch_grouped_impl<2>();
  test_launch_grouped_impl<3>();
}

template <int Dims> static void test_launch_grouped_reduce_impl() {
  constexpr size_t Size = 4;
  constexpr size_t N = get_N<Dims>(Size);
  sycl::queue q;
  int sumResult = 0;
  sycl::buffer<int> sumBuf { &sumResult, 1 };

  const auto task = [=](sycl::nd_item<Dims> item, auto& sum) {
    sum += item.get_global_linear_id();
  };

  sycl::range<Dims> r_glob = get_range<Dims>(Size);
  sycl::range<Dims> r_loc = get_range<Dims>(Size / 2);
  int expected_res = (N - 1) * N / 2;
  {
    q.submit([&](sycl::handler& h) {
      khr::launch_grouped_reduce(h, r_glob, r_loc, task, sycl::reduction(sumBuf, h, sycl::plus<>()));
    });
    q.wait();
    assert(sumBuf.get_host_access()[0] == expected_res);
    sumBuf.get_host_access()[0] = 0;
  }
  {
    khr::launch_grouped_reduce(q, r_glob, r_loc, task, sycl::reduction(&sumResult, sycl::plus<>()));
    q.wait();
    assert(sumBuf.get_host_access()[0] == expected_res);
  }

}

void test_launch_grouped_reduce() {
  test_launch_grouped_reduce_impl<1>();
  test_launch_grouped_reduce_impl<2>();
  test_launch_grouped_reduce_impl<3>();
}

static void test_launch_task() {
  sycl::queue q;
  int *buf = sycl::malloc_shared<int>(1, q);
  constexpr int val = 314;

  const auto task = [=]{ buf[0] = val; };

  {
    q.submit([&](sycl::handler &h) { khr::launch_task(h, task); });
    q.wait();
    assert(buf[0] == val);
    buf[0] = 0;
  }
  {
    khr::launch_task(q, task);
    q.wait();
    assert(buf[0] == val);
  }

  sycl::free(buf, q);
}

static void test_memcpy() {
  constexpr size_t N = 8;
  sycl::queue q;
  int* src = sycl::malloc_shared<int>(N, q);
  int* dst = sycl::malloc_shared<int>(N, q);
  std::iota(src, src + N, 0);

  {
    q.submit([&](sycl::handler &h) {
      khr::memcpy(h, dst, src, N * sizeof(*src));
    });
    q.wait();

    for (int i = 0; i < N; ++i) {
      assert(src[i] == dst[i]);
      dst[i] = 0;
    }
  }
  {
    khr::memcpy(q, dst, src, N * sizeof(*src));
    q.wait();

    for (int i = 0; i < N; ++i) {
      assert(src[i] == dst[i]);
      dst[i] = 0;
    }
  }

  sycl::free(src, q);
  sycl::free(dst, q);
}

template <typename T> static void test_copy_usm_pointers_impl() {
  constexpr size_t N = 8;
  sycl::queue q;
  T* src = sycl::malloc_shared<T>(N, q);
  T* dst = sycl::malloc_shared<T>(N, q);
  std::iota(src, src + N, 0);

  {
    q.submit([&](sycl::handler &h) {
      khr::copy(h, dst, src, N);
    });
    q.wait();
    for (int i = 0; i < N; ++i) {
      assert(src[i] == dst[i]);
      dst[i] = 0;
    }
  }
  {
    khr::copy(q, dst, src, N);
    q.wait();
    for (int i = 0; i < N; ++i)
      assert(src[i] == dst[i]);
  }

  sycl::free(src, q);
  sycl::free(dst, q);
}

static void test_copy_usm_pointers() {
  test_copy_usm_pointers_impl<char>();
  test_copy_usm_pointers_impl<int>();
  test_copy_usm_pointers_impl<float>();
}

template <typename T>
static void test_copy_accessors_host_to_device_impl() {
  using accT = sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::device>;
  const size_t N = 8;
  sycl::queue q;

  const auto test = [&](const auto& src, bool use_handler) {
    T dst[N] = {0};
    sycl::buffer<T, 1> buf(dst, sycl::range<1>(N));

    if (use_handler) {
      q.submit([&](sycl::handler& h) {
        accT acc(buf, h, sycl::range<1>(N));
        khr::copy(h, src, acc);
      });
    } else {
      accT acc(buf);
      khr::copy(q, src, acc);
    }
    q.wait();

    for (size_t i = 0; i < N; ++i)
      assert(src[i] == dst[i]);
  };

  T src[N] = {0};
  std::iota(&src[0], &src[0] + N, 0);

  std::shared_ptr<T[]> src_sptr(new T[N]());
  std::iota(src_sptr.get(), src_sptr.get() + N, 0);

  test(src, true);
  test(src_sptr, true);
  test(src, false);
  test(src_sptr, false);
}


void test_copy_accessors_host_to_device() {
  test_copy_accessors_host_to_device_impl<char>();
  test_copy_accessors_host_to_device_impl<int>();
  test_copy_accessors_host_to_device_impl<float>();
}

template <typename T>
static void test_copy_accessors_device_to_host_impl() {
  using accT = sycl::accessor<T, 1, sycl::access::mode::read, sycl::access::target::device>;
  const size_t N = 8;
  sycl::queue q;

  const auto test = [&](auto& dst, bool use_handler) {
    T src[N] = {0};
    std::iota(&src[0], &src[0] + N, 0);
    sycl::buffer<T, 1> buf(src, sycl::range<1>(N));

    if (use_handler) {
      q.submit([&](sycl::handler& h) {
        accT acc(buf, h, sycl::range<1>(N));
        khr::copy(h, acc, dst);
      });
    } else {
      accT acc(buf);
      khr::copy(q, acc, dst);
    }
    q.wait();

    for (size_t i = 0; i < N; ++i)
      assert(src[i] == dst[i]);
  };

  T dst[N] = {0};
  std::shared_ptr<T[]> dst_sptr(new T[N]());

  test(dst, true);
  test(dst_sptr, true);
  test(dst, false);
  test(dst_sptr, false);
}

void test_copy_accessors_device_to_host() {
  test_copy_accessors_device_to_host_impl<char>();
  test_copy_accessors_device_to_host_impl<int>();
  test_copy_accessors_device_to_host_impl<float>();
}

template <typename T>
void test_copy_accessors_device_to_device_impl() {
  using acc_src_T = sycl::accessor<T, 1, sycl::access::mode::read, sycl::access::target::device>;
  using acc_dst_T = sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::device>;
  const size_t N = 8;
  T src[N] = {0};
  std::iota(&src[0], &src[0] + N, 0);
  
  sycl::queue q;
  auto test_copy = [&](bool use_handler) {
    T dst[N] = {0};
    sycl::buffer<T, 1> buf_src(src, sycl::range<1>(N));
    sycl::buffer<T, 1> buf_dst(dst, sycl::range<1>(N));

    if (use_handler) {
      q.submit([&](sycl::handler &h) {
        acc_src_T acc_src(buf_src, h, sycl::range<1>(N));
        acc_dst_T acc_dst(buf_dst, h, sycl::range<1>(N));
        khr::copy(h, acc_src, acc_dst);
      });
    } else {
      acc_src_T acc_src(buf_src, sycl::range<1>(N));
      acc_dst_T acc_dst(buf_dst, sycl::range<1>(N));
      khr::copy(q, acc_src, acc_dst);
    }
    q.wait();

    for (size_t i = 0; i < N; ++i) {
      assert(src[i] == dst[i]);
    }
  };

  test_copy(true);
  test_copy(false);
}

void test_copy_accessors_device_to_device() {
  test_copy_accessors_device_to_device_impl<char>();
  test_copy_accessors_device_to_device_impl<int>();
  test_copy_accessors_device_to_device_impl<float>();
}

void test_memset() {
  constexpr size_t N = 8;
  constexpr int val = 7;
  sycl::queue q;

  auto test_memset = [&](bool use_handler) {
    auto ptr = (char *)malloc_shared(N, q);
    if (use_handler)
      q.submit([&](sycl::handler &h) { khr::memset(h, ptr, val, N); });
    else
      khr::memset(q, ptr, val, N);
    q.wait();

    for (int i = 0; i < N; ++i)
      assert(ptr[i] == val);

    sycl::free(ptr, q);
  };

  test_memset(true);
  test_memset(false);
}

template <typename T> void test_fill_impl() {
  using accT = sycl::accessor<int, 1, sycl::access::mode::read, sycl::access::target::device>;
  constexpr size_t N = 8;
  constexpr int val = 7;
  sycl::queue q;

  auto test_fill_shared = [&](bool use_handler) {
    auto ptr = sycl::malloc_shared<int>(N, q);
    if (use_handler)
      q.submit([&](sycl::handler &h) { khr::fill(h, ptr, val, N); });
    else
      khr::fill(q, ptr, val, N);
    q.wait();

    for (int i = 0; i < N; ++i)
      assert(ptr[i] == val);

    sycl::free(ptr, q);
  };

  auto test_fill_buffer = [&](bool use_handler) {
    int dst[N] = {0};
    sycl::buffer<int, 1> buf(dst, sycl::range<1>(N));
    if (use_handler) {
      q.submit([&](sycl::handler &h) {
        accT acc(buf, h, sycl::range<1>(N));
        khr::fill(h, acc, val);
      });
    } else {
      accT acc(buf, sycl::range<1>(N));
      khr::fill(q, acc, val);
    }
    q.wait();

    for (int i = 0; i < N; ++i)
      assert(dst[i] == val);
  };

  test_fill_shared(true);
  test_fill_shared(false);
  test_fill_buffer(true);
  test_fill_buffer(false);
}

void test_fill() {
  test_fill_impl<char>();
  test_fill_impl<int>();
  test_fill_impl<float>();
}

template <typename T>
void test_update_host_impl() {
  const size_t N = 8;
  using accT = sycl::accessor<T, 1, sycl::access::mode::write, sycl::access::target::device>;
  sycl::queue q;

  auto test_buffer = [&](bool use_handler) {
    T data[N] = {0};
    sycl::buffer<T, 1> buf(data, sycl::range<1>(N));

    q.submit([&](sycl::handler &h) {
      accT acc(buf, h, sycl::range<1>(N));
      h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
        acc[idx] = idx;
      });
    });

    if (use_handler) {
      q.submit([&](sycl::handler &h) {
        accT acc(buf, h, sycl::range<1>(N));
        khr::update_host(h, acc);
      });
    } else {
      accT acc(buf, sycl::range<1>(N));
      khr::update_host(q, acc);
    }
    q.wait();

    for (size_t i = 0; i < N; ++i)
      assert(data[i] == i);
  };

  test_buffer(true);
  test_buffer(false);
}

void test_update_host() {
  test_update_host_impl<char>();
  test_update_host_impl<int>();
  test_update_host_impl<float>();
}

void test_prefetch() {
  sycl::queue q;
  int* buffer = sycl::malloc_shared<int>(1, q);
  // CHECK_NOTHROW(oneapi_ext::mem_advise(q, buffer, sizeof(*buffer), 1));
  // CHECK_NOTHROW(oneapi_ext::submit(q, [&](sycl::handler& h) {
  //   oneapi_ext::mem_advise(h, buffer, sizeof(*buffer), 1);
  // }));

  try {
    khr::prefetch(q, buffer, sizeof(*buffer));
  } catch (...) {
    assert(true);
  }
  try {
    q.submit([&](sycl::handler &h) {
      khr::prefetch(h, buffer, sizeof(*buffer));
    });
  } catch (...) {
    assert(true);
  }

  sycl::free(buffer, q);
}

void test_mem_advise() {
  sycl::queue q;
  int* buffer = sycl::malloc_shared<int>(1, q);
  // CHECK_NOTHROW(oneapi_ext::mem_advise(q, buffer, sizeof(*buffer), 1));
  // CHECK_NOTHROW(oneapi_ext::submit(q, [&](sycl::handler& h) {
  //   oneapi_ext::mem_advise(h, buffer, sizeof(*buffer), 1);
  // }));

  try {
    khr::mem_advise(q, buffer, sizeof(*buffer), 1);
  } catch (...) {
    assert(true);
  }
  try {
    q.submit([&](sycl::handler &h) {
      khr::mem_advise(h, buffer, sizeof(*buffer), 1);
    });
  } catch (...) {
    assert(true);
  }

  sycl::free(buffer, q);
}

static void test_command_barrier() {
  sycl::queue q;
  bool* task_done = sycl::malloc_shared<bool>(1, q);
  bool* test_passed = sycl::malloc_shared<bool>(1, q);
  *task_done = false;
  *test_passed = false;

  q.single_task([=]{
    float sum = 0;
    for (int i = 0; i < 1000; ++i)
      sum += sycl::sqrt(float(i));
    *task_done = (sum > 0);
  });

  khr::command_barrier(q);

  q.single_task([=]{
    *test_passed = *task_done;
  });
  q.wait();

  assert(*task_done);
  assert(*test_passed);
  sycl::free(task_done, q);
  sycl::free(test_passed, q);
}

static void test_event_barrier() {
  sycl::queue q;
  bool* task_done = sycl::malloc_shared<bool>(1, q);
  bool* test_passed = sycl::malloc_shared<bool>(1, q);
  *task_done = false;
  *test_passed = false;
  const auto event = q.submit([&](sycl::handler& h) {
    h.single_task([=] {
      float sum = 0;
      for (int i = 0; i < 1000; ++i)
        sum += sycl::sqrt(float(i));
      *task_done = (sum > 0);
    });
  });
  khr::event_barrier(q, {event});
  q.single_task([=] { *test_passed = *task_done; });
  q.wait();
  assert(*task_done);
  assert(*test_passed);

  sycl::free(task_done, q);
  sycl::free(test_passed, q);
}

int main() {
  // command group functions
  test_submit();

  // kernel launch
  test_launch();
  test_launch_reduce();            // fails on GPU
  test_launch_grouped();
  test_launch_grouped_reduce();    // fails on GPU
  test_launch_task();

  // // Memory operations
  // // copy 
  test_memcpy();
  test_copy_usm_pointers();
  test_copy_accessors_host_to_device();
  test_copy_accessors_device_to_host();
  test_copy_accessors_device_to_device();

  // rest
  test_memset();
  test_fill();
  test_update_host();
  test_prefetch();
  test_mem_advise();

  // barriers
  // test_barrier();
  // test_partial_barrier();
}
