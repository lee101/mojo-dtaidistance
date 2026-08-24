"""DTW kernels and their C ABI in one compilation unit."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def inf() -> Float64:
    var zero = 0.0
    return 1.0 / zero


def fill_f64(values: Ptr, start: Int, count: Int, value: Float64):
    comptime W = simd_width_of[DType.float64]()
    var vector = SIMD[DType.float64, W](value)
    var i = 0
    while i + W <= count:
        values.store(start + i, vector)
        i += W
    while i < count:
        values[start + i] = value
        i += 1


def point_cost(
    a: Ptr, ai: Int, b: Ptr, bi: Int, ndim: Int, inner: Int
) -> Float64:
    var total = 0.0
    var abase = ai * ndim
    var bbase = bi * ndim
    for d in range(ndim):
        var delta = a[abase + d] - b[bbase + d]
        total += delta * delta
    if inner == 1:
        return sqrt(total)
    return total


def adjusted(value: Float64, inner: Int) -> Float64:
    if value <= 0.0:
        return inf()
    if inner == 0:
        return value * value
    return value


def adjusted_penalty(value: Float64, inner: Int) -> Float64:
    if value <= 0.0:
        return 0.0
    if inner == 0:
        return value * value
    return value


def finish(value: Float64, inner: Int) -> Float64:
    if inner == 0:
        return sqrt(value)
    return value


def dtw_distance_squared_default(
    a: Ptr,
    b: Ptr,
    n: Int,
    m: Int,
    window_in: Int,
    work: Ptr,
) -> Float64:
    var window = window_in
    if window <= 0:
        window = max(n, m)
    var width = m + 1
    fill_f64(work, 0, 2 * width, inf())
    work[0] = 0.0
    var previous = 0
    var current = width
    for i in range(n):
        var j_start = max(0, i - max(0, n - m) - window + 1)
        var j_end = min(m, i + max(0, m - n) + window)
        work[current + j_start] = inf()
        for j in range(j_start, j_end):
            var delta = a[i] - b[j]
            var best = work[previous + j]
            var vertical = work[previous + j + 1]
            if vertical < best:
                best = vertical
            var horizontal = work[current + j]
            if horizontal < best:
                best = horizontal
            work[current + j + 1] = delta * delta + best
        if i + 1 < n:
            var swap = previous
            previous = current
            current = swap
    return sqrt(work[current + m])


def dtw_distance(
    a: Ptr,
    b: Ptr,
    n: Int,
    m: Int,
    ndim: Int,
    window_in: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    work: Ptr,
) -> Float64:
    if n == 0 or m == 0:
        return inf()
    if (
        ndim == 1
        and max_dist <= 0.0
        and max_step <= 0.0
        and penalty <= 0.0
        and psi_1b == 0
        and psi_1e == 0
        and psi_2b == 0
        and psi_2e == 0
        and inner == 0
    ):
        return dtw_distance_squared_default(a, b, n, m, window_in, work)
    var window = window_in
    if window <= 0:
        window = max(n, m)
    var limit = adjusted(max_dist, inner)
    var step_limit = adjusted(max_step, inner)
    var pen = adjusted_penalty(penalty, inner)
    var width = m + 1
    fill_f64(work, 0, 2 * width, inf())
    for j in range(min(m, psi_2b) + 1):
        work[j] = 0.0
    var previous = 0
    var current = width
    var end_relaxed = inf()
    for i in range(n):
        var j_start = max(0, i - max(0, n - m) - window + 1)
        var j_end = min(m, i + max(0, m - n) + window)
        fill_f64(
            work,
            current + j_start,
            min(m, j_end + 1) + 1 - j_start,
            inf(),
        )
        if i < psi_1b:
            work[current] = 0.0
        for j in range(j_start, j_end):
            var cost = point_cost(a, i, b, j, ndim, inner)
            if cost > step_limit:
                continue
            var best = work[previous + j]
            var vertical = work[previous + j + 1] + pen
            if vertical < best:
                best = vertical
            var horizontal = work[current + j] + pen
            if horizontal < best:
                best = horizontal
            var value = cost + best
            if value <= limit:
                work[current + j + 1] = value
        if j_end == m and n - 1 - i <= psi_1e:
            if work[current + m] < end_relaxed:
                end_relaxed = work[current + m]
        if i + 1 < n:
            var swap = previous
            previous = current
            current = swap
    var value = work[current + m]
    if psi_2e > 0:
        for k in range(min(m, psi_2e) + 1):
            if work[current + m - k] < value:
                value = work[current + m - k]
    if psi_1e > 0 and end_relaxed < value:
        value = end_relaxed
    if value > limit:
        return inf()
    return finish(value, inner)


def dtw_paths(
    a: Ptr,
    b: Ptr,
    n: Int,
    m: Int,
    ndim: Int,
    window_in: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    keep_internal: Int,
    psi_neg: Int,
    paths: Ptr,
) -> Float64:
    var rows = n + 1
    var cols = m + 1
    for k in range(rows * cols):
        paths[k] = inf()
    if n == 0 or m == 0:
        return inf()
    var window = window_in
    if window <= 0:
        window = max(n, m)
    var limit = adjusted(max_dist, inner)
    var step_limit = adjusted(max_step, inner)
    var pen = adjusted_penalty(penalty, inner)
    for j in range(min(m, psi_2b) + 1):
        paths[j] = 0.0
    for i in range(min(n, psi_1b) + 1):
        paths[i * cols] = 0.0
    var sc = 0
    var ec = 0
    for i in range(n):
        var j_start = max(0, i - max(0, n - m) - window + 1)
        var j_end = min(m, i + max(0, m - n) + window)
        if sc > j_start:
            j_start = sc
        var smaller_found = False
        var ec_next = i
        for j in range(j_start, j_end):
            var cost = point_cost(a, i, b, j, ndim, inner)
            if cost > step_limit:
                continue
            var best = paths[i * cols + j]
            var vertical = paths[i * cols + j + 1] + pen
            if vertical < best:
                best = vertical
            var horizontal = paths[(i + 1) * cols + j] + pen
            if horizontal < best:
                best = horizontal
            var value = cost + best
            paths[(i + 1) * cols + j + 1] = value
            if value > limit:
                if not smaller_found:
                    sc = j + 1
                if j >= ec:
                    break
            else:
                smaller_found = True
                ec_next = j + 1
        ec = ec_next
    var best_i = n
    var best_j = m
    var value = paths[n * cols + m]
    if psi_1e > 0:
        # Upstream excludes row/column zero from end relaxation.
        for k in range(min(n - 1, psi_1e) + 1):
            var candidate = paths[(n - k) * cols + m]
            if candidate < value:
                value = candidate
                best_i = n - k
                best_j = m
    if psi_2e > 0:
        for k in range(min(m - 1, psi_2e) + 1):
            var candidate = paths[n * cols + m - k]
            if candidate <= value:
                value = candidate
                best_i = n
                best_j = m - k
    if keep_internal == 0:
        if inner == 0:
            for k in range(rows * cols):
                if paths[k] >= 0.0:
                    paths[k] = sqrt(paths[k])
        value = finish(value, inner)
    if psi_neg != 0:
        if best_i < n:
            for i in range(best_i + 1, n + 1):
                paths[i * cols + m] = -1.0
        elif best_j < m:
            for j in range(best_j + 1, m + 1):
                paths[n * cols + j] = -1.0
    if value > max_dist and max_dist > 0.0 and keep_internal == 0:
        return inf()
    if keep_internal != 0 and value > limit:
        return inf()
    return value


def lb_keogh_kernel(
    a: Ptr, b: Ptr, n: Int, m: Int, ndim: Int, window_in: Int, inner: Int
) -> Float64:
    if n == 0 or m == 0:
        return inf()
    var window = window_in
    if window <= 0:
        window = max(n, m)
    var imin_diff = max(0, n - m) + window - 1
    var imax_diff = max(0, m - n) + window
    var total = 0.0
    for i in range(n):
        var lo_idx = max(0, i - imin_diff)
        var hi_idx = min(m, i + imax_diff)
        if ndim == 1:
            var lo = b[lo_idx]
            var hi = lo
            for j in range(lo_idx + 1, hi_idx):
                lo = min(lo, b[j])
                hi = max(hi, b[j])
            if a[i] > hi:
                var delta = a[i] - hi
                total += delta * delta if inner == 0 else abs(delta)
            elif a[i] < lo:
                var delta = a[i] - lo
                total += delta * delta if inner == 0 else abs(delta)
        else:
            var best = inf()
            for j in range(lo_idx, hi_idx):
                var candidate = point_cost(a, i, b, j, ndim, inner)
                if candidate < best:
                    best = candidate
            total += best
    return finish(total, inner)


def euclidean_kernel(
    a: Ptr, b: Ptr, n: Int, m: Int, ndim: Int, inner: Int
) -> Float64:
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return inf()
    if n == m and inner == 0:
        comptime W = simd_width_of[DType.float64]()
        var vector_total = SIMD[DType.float64, W](0.0)
        var total_count = n * ndim
        var k = 0
        while k + W <= total_count:
            var delta = a.load[width=W](k) - b.load[width=W](k)
            vector_total += delta * delta
            k += W
        var total = vector_total.reduce_add()
        while k < total_count:
            var delta = a[k] - b[k]
            total += delta * delta
            k += 1
        return sqrt(total)
    var common = min(n, m)
    var total = 0.0
    for i in range(common):
        total += point_cost(a, i, b, i, ndim, inner)
    if n > m:
        for i in range(m, n):
            total += point_cost(a, i, b, m - 1, ndim, inner)
    elif m > n:
        for j in range(n, m):
            total += point_cost(a, n - 1, b, j, ndim, inner)
    return finish(total, inner)


def distance_pair_at(
    data: Ptr,
    offsets: IPtr,
    pairs: IPtr,
    pidx: Int,
    ndim: Int,
    window: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    work: Ptr,
    result: Ptr,
):
    var ai = Int(pairs[2 * pidx])
    var bi = Int(pairs[2 * pidx + 1])
    var a0 = Int(offsets[ai])
    var b0 = Int(offsets[bi])
    var n = Int(offsets[ai + 1]) - a0
    var m = Int(offsets[bi + 1]) - b0
    result[pidx] = dtw_distance(
        data + a0 * ndim,
        data + b0 * ndim,
        n,
        m,
        ndim,
        window,
        max_dist,
        max_step,
        penalty,
        psi_1b,
        psi_1e,
        psi_2b,
        psi_2e,
        inner,
        work,
    )


def distance_pairs(
    data: Ptr,
    offsets: IPtr,
    pairs: IPtr,
    count: Int,
    ndim: Int,
    window: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    use_parallel: Int,
    work_stride: Int,
    work: Ptr,
    result: Ptr,
):
    # Mojo 1.1 moved the parallel task runtime out of the standalone standard
    # library. Keep the C/Python ABI (including use_parallel and work_stride)
    # stable and use the serial kernel until a public standalone replacement is
    # available.
    for pidx in range(count):
        distance_pair_at(
            data,
            offsets,
            pairs,
            pidx,
            ndim,
            window,
            max_dist,
            max_step,
            penalty,
            psi_1b,
            psi_1e,
            psi_2b,
            psi_2e,
            inner,
            work,
            result,
        )


def assign_clusters(
    data: Ptr,
    offsets: IPtr,
    count: Int,
    centers: Ptr,
    center_offsets: IPtr,
    k: Int,
    ndim: Int,
    window: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    work: Ptr,
    labels: IPtr,
    distances: Ptr,
):
    for idx in range(count):
        var a0 = Int(offsets[idx])
        var n = Int(offsets[idx + 1]) - a0
        var best = inf()
        var best_cluster = 0
        for cluster in range(k):
            var b0 = Int(center_offsets[cluster])
            var m = Int(center_offsets[cluster + 1]) - b0
            var candidate = dtw_distance(
                data + a0 * ndim,
                centers + b0 * ndim,
                n,
                m,
                ndim,
                window,
                max_dist,
                max_step,
                penalty,
                psi_1b,
                psi_1e,
                psi_2b,
                psi_2e,
                inner,
                work,
            )
            if candidate < best:
                best = candidate
                best_cluster = cluster
        labels[idx] = Int64(best_cluster)
        distances[idx] = best


def dba_update(
    data: Ptr,
    offsets: IPtr,
    count: Int,
    center: Ptr,
    center_len: Int,
    ndim: Int,
    mask: BPtr,
    window: Int,
    max_dist: Float64,
    max_step: Float64,
    penalty: Float64,
    psi_1b: Int,
    psi_1e: Int,
    psi_2b: Int,
    psi_2e: Int,
    inner: Int,
    paths: Ptr,
    sums: Ptr,
    counts: Ptr,
):
    for i in range(center_len * ndim):
        sums[i] = 0.0
    for i in range(center_len):
        counts[i] = 0.0
    for idx in range(count):
        if mask[idx] == 0:
            continue
        var start = Int(offsets[idx])
        var length = Int(offsets[idx + 1]) - start
        _ = dtw_paths(
            center,
            data + start * ndim,
            center_len,
            length,
            ndim,
            window,
            max_dist,
            max_step,
            penalty,
            psi_1b,
            psi_1e,
            psi_2b,
            psi_2e,
            inner,
            1,
            0,
            paths,
        )
        var cols = length + 1
        var pi = center_len
        var pj = length
        var endpoint = paths[pi * cols + pj]
        if psi_1e > 0:
            for k in range(min(center_len - 1, psi_1e) + 1):
                var candidate = paths[(center_len - k) * cols + length]
                if candidate < endpoint:
                    endpoint = candidate
                    pi = center_len - k
                    pj = length
        if psi_2e > 0:
            for k in range(min(length - 1, psi_2e) + 1):
                var candidate = paths[center_len * cols + length - k]
                if candidate <= endpoint:
                    endpoint = candidate
                    pi = center_len
                    pj = length - k
        var pen = adjusted_penalty(penalty, inner)
        while pi > 0 and pj > 0:
            counts[pi - 1] += 1.0
            for d in range(ndim):
                sums[(pi - 1) * ndim + d] += data[(start + pj - 1) * ndim + d]
            var diagonal = paths[(pi - 1) * cols + pj - 1]
            var vertical = paths[(pi - 1) * cols + pj] + pen
            var horizontal = paths[pi * cols + pj - 1] + pen
            if diagonal <= vertical and diagonal <= horizontal:
                pi -= 1
                pj -= 1
            elif vertical <= horizontal:
                pi -= 1
            else:
                pj -= 1


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


@export("mdd_distance")
def mdd_distance(
    a: Int, b: Int, n: Int, m: Int, ndim: Int, window: Int,
    max_dist: Float64, max_step: Float64, penalty: Float64,
    psi_1b: Int, psi_1e: Int, psi_2b: Int, psi_2e: Int,
    inner: Int, work: Int
) abi("C") -> Float64:
    return dtw_distance(
        p(a), p(b), n, m, ndim, window, max_dist, max_step, penalty,
        psi_1b, psi_1e, psi_2b, psi_2e, inner, p(work)
    )


@export("mdd_paths")
def mdd_paths(
    a: Int, b: Int, n: Int, m: Int, ndim: Int, window: Int,
    max_dist: Float64, max_step: Float64, penalty: Float64,
    psi_1b: Int, psi_1e: Int, psi_2b: Int, psi_2e: Int,
    inner: Int, keep_internal: Int, psi_neg: Int, paths: Int
) abi("C") -> Float64:
    return dtw_paths(
        p(a), p(b), n, m, ndim, window, max_dist, max_step, penalty,
        psi_1b, psi_1e, psi_2b, psi_2e, inner, keep_internal, psi_neg, p(paths)
    )


@export("mdd_lb_keogh")
def mdd_lb_keogh(
    a: Int, b: Int, n: Int, m: Int, ndim: Int, window: Int, inner: Int
) abi("C") -> Float64:
    return lb_keogh_kernel(p(a), p(b), n, m, ndim, window, inner)


@export("mdd_euclidean")
def mdd_euclidean(
    a: Int, b: Int, n: Int, m: Int, ndim: Int, inner: Int
) abi("C") -> Float64:
    return euclidean_kernel(p(a), p(b), n, m, ndim, inner)


@export("mdd_distance_pairs")
def mdd_distance_pairs(
    data: Int, offsets: Int, pairs: Int, count: Int, ndim: Int, window: Int,
    max_dist: Float64, max_step: Float64, penalty: Float64,
    psi_1b: Int, psi_1e: Int, psi_2b: Int, psi_2e: Int,
    inner: Int, use_parallel: Int, work_stride: Int, work: Int, result: Int
) abi("C"):
    distance_pairs(
        p(data), ip(offsets), ip(pairs), count, ndim, window, max_dist,
        max_step, penalty, psi_1b, psi_1e, psi_2b, psi_2e, inner,
        use_parallel, work_stride, p(work), p(result)
    )


@export("mdd_assign")
def mdd_assign(
    data: Int, offsets: Int, count: Int, centers: Int, center_offsets: Int,
    k: Int, ndim: Int, window: Int, max_dist: Float64, max_step: Float64,
    penalty: Float64, psi_1b: Int, psi_1e: Int, psi_2b: Int, psi_2e: Int,
    inner: Int, work: Int, labels: Int, distances: Int
) abi("C"):
    assign_clusters(
        p(data), ip(offsets), count, p(centers), ip(center_offsets), k, ndim,
        window, max_dist, max_step, penalty, psi_1b, psi_1e, psi_2b, psi_2e,
        inner, p(work), ip(labels), p(distances)
    )


@export("mdd_dba_update")
def mdd_dba_update(
    data: Int, offsets: Int, count: Int, center: Int, center_len: Int,
    ndim: Int, mask: Int, window: Int, max_dist: Float64,
    max_step: Float64, penalty: Float64, psi_1b: Int, psi_1e: Int,
    psi_2b: Int, psi_2e: Int, inner: Int, paths: Int, sums: Int, counts: Int
) abi("C"):
    dba_update(
        p(data), ip(offsets), count, p(center), center_len, ndim, bp(mask),
        window, max_dist, max_step, penalty, psi_1b, psi_1e, psi_2b, psi_2e,
        inner, p(paths), p(sums), p(counts)
    )
