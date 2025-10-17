# -*- coding: utf-8 -*-


import json, re, math

# ---------- regex ----------
_PAT_OUT   = re.compile(r"<output>\s*(.*?)\s*</output>", re.S | re.IGNORECASE)
_PAT_THINK = re.compile(r"<think>\s*.*?\s*</think>", re.S | re.IGNORECASE)

# ---------- parse helpers ----------
def _extract_output(block: str):
    """Return parsed JSON inside <output>...</output>, or None."""
    if not block:
        return None
    m = _PAT_OUT.search(block)
    if not m:
        return None
    s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

def _has_both_tags(solution_str: str) -> bool:
    """Require BOTH a <think>...</think> and a JSON-parsable <output>...</output>."""
    if not solution_str:
        return False
    has_think = bool(_PAT_THINK.search(solution_str))
    has_output_json = _extract_output(solution_str) is not None
    return has_think and has_output_json

# ---------- common small utils ----------
EPS = 1e-12
SQRT2 = 2 ** 0.5

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _l2(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _norm(v):   return math.hypot(v[0], v[1])

def _unit(v):
    n = _norm(v)
    return (0.0, 0.0) if n < EPS else (v[0]/n, v[1]/n)

def _angle_between(u, v):
    un, vn = _norm(u), _norm(v)
    if un < EPS or vn < EPS:
        return 0.0
    dot = max(-1.0, min(1.0, (u[0]*v[0] + u[1]*v[1]) / (un * vn)))
    return math.acos(dot)  # [0, pi]

# ---------- bbox: center (x,y), width, height -> xyxy ----------
def _xywhc_to_xyxy(box):
    x = float(box["x"]); y = float(box["y"])
    w = abs(float(box["width"])); h = abs(float(box["height"]))
    x1 = _clip01(x - w/2.0); y1 = _clip01(y - h/2.0)
    x2 = _clip01(x + w/2.0); y2 = _clip01(y + h/2.0)
    return x1, y1, x2, y2

def _area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou_xyxy(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter = _area(inter_x1, inter_y1, inter_x2, inter_y2)
    a1, a2 = _area(x1, y1, x2, y2), _area(X1, Y1, X2, Y2)
    denom = a1 + a2 - inter
    if denom <= 0:
        return 0.0
    return inter / denom

def _giou_xyxy(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    iou = _iou_xyxy(b1, b2)
    c_x1, c_y1 = min(x1, X1), min(y1, Y1)
    c_x2, c_y2 = max(x2, X2), max(y2, Y2)
    c_area = _area(c_x1, c_y1, c_x2, c_y2) or 1e-12
    inter_x1, inter_y1 = max(x1, X1), max(y1, Y1)
    inter_x2, inter_y2 = min(x2, X2), min(y2, Y2)
    inter = _area(inter_x1, inter_y1, inter_x2, inter_y2)
    a1, a2 = _area(x1, y1, x2, y2), _area(X1, Y1, X2, Y2)
    union = max(1e-12, a1 + a2 - inter)
    giou = iou - (c_area - union) / c_area
    return max(-1.0, min(1.0, giou)), iou

# ---------- traj: orientation-sensitive discrete Fréchet (no resampling) ----------
def _local_dir(points, i):
    n = len(points)
    if n == 1: return (0.0, 0.0)
    if i == 0: return _unit(_sub(points[1], points[0]))
    if i == n-1: return _unit(_sub(points[-1], points[-2]))
    d1 = _unit(_sub(points[i], points[i-1]))
    d2 = _unit(_sub(points[i+1], points[i]))
    s = (d1[0] + d2[0], d1[1] + d2[1])
    nrm = _norm(s)
    return (s[0]/nrm, s[1]/nrm) if nrm >= EPS else d1

def _local_len(points, i):
    n = len(points)
    if n <= 1: return 0.0
    if i < n-1: return _norm(_sub(points[i+1], points[i]))
    return _norm(_sub(points[i], points[i-1]))

def _os_discrete_frechet(P, Q, lambda_theta=0.25, lambda_ratio=0.10):

    m, n = len(P), len(Q)
    tP = [_local_dir(P, i) for i in range(m)]
    tQ = [_local_dir(Q, j) for j in range(n)]
    lP = [_local_len(P, i) for i in range(m)]
    lQ = [_local_len(Q, j) for j in range(n)]

    def cost(i, j):
        euc = _l2(P[i], Q[j])
        ang = _angle_between(tP[i], tQ[j])
        if lP[i] > EPS and lQ[j] > EPS:
            rat = abs(math.log(lP[i] / lQ[j]))
        else:
            rat = abs(math.log((lP[i] + EPS) / (lQ[j] + EPS)))
        return euc + lambda_theta * ang + lambda_ratio * rat

    ca = [[-1.0]*n for _ in range(m)]
    def C(i, j):
        if ca[i][j] > -0.5: return ca[i][j]
        c = cost(i, j)
        if i == 0 and j == 0:
            val = c
        elif i > 0 and j == 0:
            val = max(C(i-1, 0), c)
        elif i == 0 and j > 0:
            val = max(C(0, j-1), c)
        else:
            val = max(min(C(i-1, j), C(i-1, j-1), C(i, j-1)), c)
        ca[i][j] = val
        return val

    return C(m-1, n-1)

def _discrete_frechet(P, Q):

    m, n = len(P), len(Q)
    ca = [[-1.0]*n for _ in range(m)]
    def C(i, j):
        if ca[i][j] > -0.5: return ca[i][j]
        d = _l2(P[i], Q[j])
        if i == 0 and j == 0:
            val = d
        elif i > 0 and j == 0:
            val = max(C(i-1, 0), d)
        elif i == 0 and j > 0:
            val = max(C(0, j-1), d)
        else:
            val = max(min(C(i-1, j), C(i-1, j-1), C(i, j-1)), d)
        ca[i][j] = val
        return val
    return C(m-1, n-1)

# ---------- type checks ----------
def _is_bbox(obj):
    return isinstance(obj, dict) and all(k in obj for k in ("x", "y", "width", "height"))

def _is_traj2d(obj):
    return isinstance(obj, list) and all(isinstance(t, (list, tuple)) and len(t) == 2 for t in obj)

def _parse_traj(obj):
    pts = [(_clip01(float(x)), _clip01(float(y))) for x, y in obj]
    return pts

# ---------- main entry (VERL) ----------
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Returns FIXED keys for all samples:
      - score, task_score, format_score
      - giou, iou  (bbox diag; NaN for traj)
      - os_frechet, os_frechet_norm, os_fre_score, (traj; NaN for bbox)
      - dfd (traj diag; NaN for bbox)
      - mode ("bbox"/"traj") for quick inspection
    """
    extra_info = extra_info or {}
    lambda_theta = float(extra_info.get("lambda_theta", 0.25))
    lambda_ratio = float(extra_info.get("lambda_ratio", 0.10))
    alpha_fre    = float(extra_info.get("alpha_fre", 1.0))
    alpha_end    = float(extra_info.get("alpha_end", 1.0))

    # 1) parse pred & gt payloads from <output>...</output>
    pred = _extract_output(solution_str)
    gt   = _extract_output(ground_truth)

    # 2) format reward
    format_score = 1.0 if _has_both_tags(solution_str) else 0.0

    # 3) init fixed outputs
    task_score = 0.0
    giou = float("nan")
    iou  = float("nan")
    os_frechet = float("nan")
    os_frechet_norm = float("nan")
    os_fre_score = float("nan")
    end_dist = float("nan")
    end_norm = float("nan")
    end_score = float("nan")
    dfd = float("nan")
    mode = "none"

    try:
        
        # -------- bbox branch --------
        if _is_bbox(pred) and _is_bbox(gt):
            mode = "bbox"
            b1 = _xywhc_to_xyxy(pred)
            b2 = _xywhc_to_xyxy(gt)
            giou, iou = _giou_xyxy(b1, b2)
            task_score = (giou + 1.0) / 2.0  # [-1,1] -> [0,1]

        # -------- trajectory branch --------
        elif _is_traj2d(pred) and _is_traj2d(gt):
            mode = "traj"
            P = _parse_traj(gt)
            Q = _parse_traj(pred)

            # OS-DFD (strict, no resampling)
            os_raw = _os_discrete_frechet(P, Q, lambda_theta=lambda_theta, lambda_ratio=lambda_ratio)

            # conservative normalizer: Euclidean <= sqrt(2), angle <= pi, |log len_ratio| <= log(10) (cap)
            OS_DENOM = SQRT2 + lambda_theta * math.pi + lambda_ratio * math.log(10.0)
            os_n = min(1.0, os_raw / OS_DENOM)
            os_score = 1.0 - os_n

            # Standard DFD (diagnostic only)
            dfd = _discrete_frechet(P, Q)/SQRT2

            # write out
            os_frechet_norm, os_fre_score = float(os_n), float(os_score)

            task_score = os_score 

    except:
        print(f"Failed to compute reward")
        task_score=0


    # 4) final score
    final = 0.9 * task_score+ 0.1 * format_score

    # 5) return with fixed keys
    return {
        "score": float(final),

        "iou": float(iou),

        # trajectory metrics
        "os_frechet_norm": float(os_frechet_norm),
    }

