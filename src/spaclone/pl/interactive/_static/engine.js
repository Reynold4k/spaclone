/*! spaclone interactive figure engine 1.1.0 | part of spaclone (MIT licence, (c) Chen Zhu) |
 * https://github.com/Reynold4k/spaclone | The data are inside this file; nothing is loaded from the network.
 * Drawing is done on the CPU into a 2D canvas (no WebGL), so the figure works wherever a canvas does. */
(function (global) {
  "use strict";
  var VERSION = "1.1.0";
  var engines = (global.__spacloneEngines = global.__spacloneEngines || {});
  var SCRIPT = global.document && global.document.currentScript;  // the script element running this code
  if (engines[VERSION]) { engines[VERSION].mountAll(SCRIPT); return; }

  // ------------------------------------------------------------------------------------------------ small helpers
  var DOC = global.document;
  function el(tag, cls, parent, text) {
    var e = DOC.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function isNum(v) { return typeof v === "number" && v === v && v !== Infinity && v !== -Infinity; }
  function matches(when, state) {
    if (!when) return true;
    for (var k in when) {
      if (!Object.prototype.hasOwnProperty.call(when, k)) continue;
      var want = when[k], have = state[k];
      if (Array.isArray(want)) { if (want.indexOf(have) < 0) return false; } else if (want !== have) return false;
    }
    return true;
  }
  var NF = (typeof Intl !== "undefined" && Intl.NumberFormat) ? new Intl.NumberFormat("en-US") : null;
  function fmtInt(v) { v = Math.round(v); return NF ? NF.format(v) : String(v); }
  function fmtNum(v, f) {
    if (v === null || v === undefined || (typeof v === "number" && v !== v)) return "\u2013";
    if (typeof v !== "number") return String(v);
    if (f === "pct") return (Math.abs(v) < 0.1 && v !== 0 ? (100 * v).toPrecision(2) : (100 * v).toFixed(1).replace(/\.0$/, "")) + "%";
    if (f === "pct0") return Math.round(100 * v) + "%";
    if (f === "int") return fmtInt(v);
    if (f === "p") return v < 1e-3 ? v.toExponential(1) : v.toPrecision(2);
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (Number.isInteger(v) && a < 1e7) return fmtInt(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(2).replace("e+", "e");
    var s = v.toPrecision(3);
    if (s.indexOf("e") < 0 && s.indexOf(".") >= 0) s = s.replace(/0+$/, "").replace(/\.$/, "");
    return s;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function hexRgb(h) {
    h = String(h || "#000000").replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var v = parseInt(h.slice(0, 6), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  function u32(h, alpha) {  // little-endian RGBA packed for a Uint32 view of ImageData
    var c = hexRgb(h), a = alpha === undefined ? 255 : Math.round(255 * alpha);
    return ((a << 24) | (c[2] << 16) | (c[1] << 8) | c[0]) >>> 0;
  }
  function rgba(h, a) { var c = hexRgb(h); return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")"; }
  function rampLut(stops) {  // 256 colours interpolated linearly in sRGB between equally spaced stops
    var cs = stops.map(hexRgb), lut = new Uint32Array(256), n = cs.length;
    for (var i = 0; i < 256; i++) {
      var t = i / 255 * (n - 1), k = Math.min(Math.floor(t), n - 2), f = t - k;
      if (n === 1) { k = 0; f = 0; }
      var a = cs[k], b = cs[Math.min(k + 1, n - 1)];
      lut[i] = ((255 << 24) | (Math.round(a[2] + (b[2] - a[2]) * f) << 16) | (Math.round(a[1] + (b[1] - a[1]) * f) << 8) |
        Math.round(a[0] + (b[0] - a[0]) * f)) >>> 0;
    }
    return lut;
  }
  function rampCss(stops) { return "linear-gradient(90deg," + stops.join(",") + ")"; }
  function lutHex(lut, i) { var v = lut[i]; return "#" + ((1 << 24) | ((v & 255) << 16) | (((v >> 8) & 255) << 8) | ((v >> 16) & 255)).toString(16).slice(1); }
  function niceStep(span, n) {
    var raw = span / Math.max(n, 1), p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }
  function linTicks(lo, hi, n) {
    if (!(hi > lo)) return [lo];
    var st = niceStep(hi - lo, n), out = [], v = Math.ceil(lo / st - 1e-9) * st;
    for (; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
    return out;
  }
  function logTicks(lo, hi) {  // lo, hi in log10 units
    var out = [], a = Math.floor(lo), b = Math.ceil(hi), subs = (b - a) <= 2 ? [1, 2, 5] : (b - a) <= 5 ? [1, 3] : [1];
    if (b - a > 8) { var st = Math.ceil((b - a) / 6); for (var e = a; e <= b; e += st) if (e >= lo - 1e-9 && e <= hi + 1e-9) out.push(e); return out; }
    for (var e2 = a; e2 <= b; e2++) for (var s = 0; s < subs.length; s++) {
      var v = e2 + Math.log10(subs[s]); if (v >= lo - 1e-9 && v <= hi + 1e-9) out.push(v);
    }
    return out;
  }
  function niceLength(target) {  // 1, 2 or 5 x 10^k closest below-ish the target
    if (!(target > 0)) return 0;
    var p = Math.pow(10, Math.floor(Math.log10(target))), best = p;
    [1, 2, 2.5, 5, 10].forEach(function (m) { if (m * p <= target * 1.15) best = m * p; });
    return best;
  }
  function fmtLen(um) { return um >= 1000 ? (um / 1000) + " mm" : (um >= 1 ? um : um.toPrecision(2)) + " \u00b5m"; }
  function pip(px, py, xs, ys) {  // even-odd crossing test; the same rule as spaclone.pl.interactive (Python)
    var inside = false, n = xs.length;
    for (var i = 0, j = n - 1; i < n; j = i++) {
      var yi = ys[i], yj = ys[j];
      if ((yi > py) !== (yj > py)) {
        var xc = (xs[j] - xs[i]) * (py - yi) / (yj - yi) + xs[i];
        if (px < xc) inside = !inside;
      }
    }
    return inside;
  }

  // ------------------------------------------------------------------------------------------------ decoding
  var B64 = new Int16Array(256).fill(-1);
  (function () { var s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; for (var i = 0; i < 64; i++) B64[s.charCodeAt(i)] = i; })();
  function b64bytes(s) {
    var n = s.length; while (n > 0 && s.charCodeAt(n - 1) === 61) n--;
    var out = new Uint8Array((n * 3) >> 2), o = 0, i = 0;
    for (; i + 4 <= n; i += 4) {
      var v = (B64[s.charCodeAt(i)] << 18) | (B64[s.charCodeAt(i + 1)] << 12) | (B64[s.charCodeAt(i + 2)] << 6) | B64[s.charCodeAt(i + 3)];
      out[o++] = (v >> 16) & 255; out[o++] = (v >> 8) & 255; out[o++] = v & 255;
    }
    var r = n - i;
    if (r === 2) { var w = (B64[s.charCodeAt(i)] << 18) | (B64[s.charCodeAt(i + 1)] << 12); out[o++] = (w >> 16) & 255; }
    else if (r === 3) { var x = (B64[s.charCodeAt(i)] << 18) | (B64[s.charCodeAt(i + 1)] << 12) | (B64[s.charCodeAt(i + 2)] << 6); out[o++] = (x >> 16) & 255; out[o++] = (x >> 8) & 255; }
    return out;
  }
  // DEFLATE (RFC 1951) inside a zlib wrapper (RFC 1950): written for spaclone from the RFCs.
  var LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  var LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  var DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  var DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  var CLORD = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
  function huff(lengths, n) {
    var max = 0, i;
    for (i = 0; i < n; i++) if (lengths[i] > max) max = lengths[i];
    if (max === 0) return { t: new Uint32Array(1), m: 0 };
    var cnt = new Uint16Array(16), next = new Uint16Array(16), code = 0;
    for (i = 0; i < n; i++) cnt[lengths[i]]++;
    cnt[0] = 0;
    for (var b = 1; b <= 15; b++) { code = (code + cnt[b - 1]) << 1; next[b] = code; }
    var size = 1 << max, t = new Uint32Array(size);
    for (var s = 0; s < n; s++) {
      var L = lengths[s]; if (!L) continue;
      var c = next[L]++, r = 0;
      for (var k = 0; k < L; k++) r |= ((c >> k) & 1) << (L - 1 - k);
      for (var f = r; f < size; f += 1 << L) t[f] = (s << 4) | L;
    }
    return { t: t, m: max };
  }
  var FIXED = null;
  function fixedTables() {
    if (FIXED) return FIXED;
    var l = new Uint8Array(288), d = new Uint8Array(30), i;
    for (i = 0; i < 144; i++) l[i] = 8; for (; i < 256; i++) l[i] = 9; for (; i < 280; i++) l[i] = 7; for (; i < 288; i++) l[i] = 8;
    for (i = 0; i < 30; i++) d[i] = 5;
    FIXED = [huff(l, 288), huff(d, 30)];
    return FIXED;
  }
  function inflateZlib(src, rawLen) {
    if (src.length < 6 || (src[0] & 15) !== 8 || ((src[0] << 8) | src[1]) % 31 !== 0 || (src[1] & 32)) throw new Error("data block is not zlib-compressed");
    var out = new Uint8Array(rawLen > 0 ? rawLen : Math.max(1024, src.length * 4)), op = 0, pos = 2, end = src.length - 4;
    var bb = 0, bc = 0;
    function grow(need) { if (op + need <= out.length) return; var o2 = new Uint8Array(Math.max(out.length * 2, op + need)); o2.set(out.subarray(0, op)); out = o2; }
    function fill(n) { while (bc < n) { var byte = pos < end ? src[pos] : 0; if (pos > end + 4) throw new Error("compressed data ended early"); pos++; bb |= byte << bc; bc += 8; } }
    function bits(n) { if (!n) return 0; fill(n); var v = bb & ((1 << n) - 1); bb >>>= n; bc -= n; return v; }
    function sym(h) {
      fill(h.m); var e = h.t[bb & ((1 << h.m) - 1)], L = e & 15;
      if (!L) throw new Error("damaged compressed data (bad code)");
      bb >>>= L; bc -= L; return e >>> 4;
    }
    var last = 0;
    while (!last) {
      last = bits(1); var type = bits(2);
      if (type === 0) {
        bb >>>= bc & 7; bc -= bc & 7;
        var len = bits(16), nlen = bits(16);
        if ((len ^ 0xffff) !== nlen) throw new Error("damaged compressed data (stored block)");
        pos -= bc >> 3; bb = 0; bc = 0;
        grow(len); out.set(src.subarray(pos, pos + len), op); op += len; pos += len;
        continue;
      }
      var lt, dt;
      if (type === 1) { var ft = fixedTables(); lt = ft[0]; dt = ft[1]; }
      else if (type === 2) {
        var hl = bits(5) + 257, hd = bits(5) + 1, hc = bits(4) + 4, cl = new Uint8Array(19), i;
        for (i = 0; i < hc; i++) cl[CLORD[i]] = bits(3);
        var ct = huff(cl, 19), lens = new Uint8Array(hl + hd), k2 = 0;
        while (k2 < hl + hd) {
          var s = sym(ct);
          if (s < 16) lens[k2++] = s;
          else if (s === 16) { if (!k2) throw new Error("damaged compressed data (repeat)"); var pr = lens[k2 - 1], rep = 3 + bits(2); while (rep--) lens[k2++] = pr; }
          else if (s === 17) { var z = 3 + bits(3); while (z--) lens[k2++] = 0; }
          else { var z2 = 11 + bits(7); while (z2--) lens[k2++] = 0; }
        }
        lt = huff(lens.subarray(0, hl), hl); dt = huff(lens.subarray(hl), hd);
      } else throw new Error("damaged compressed data (block type)");
      for (;;) {
        var c = sym(lt);
        if (c < 256) { if (op >= out.length) grow(1); out[op++] = c; continue; }
        if (c === 256) break;
        c -= 257; if (c >= 29) throw new Error("damaged compressed data (length)");
        var ln = LBASE[c] + bits(LEXT[c]), dc = sym(dt);
        if (dc >= 30) throw new Error("damaged compressed data (distance)");
        var dist = DBASE[dc] + bits(DEXT[dc]);
        if (dist > op) throw new Error("damaged compressed data (distance too far)");
        grow(ln);
        for (var q = 0; q < ln; q++) { out[op] = out[op - dist]; op++; }
      }
    }
    var res = op === out.length ? out : out.subarray(0, op);
    var a = 1, b2 = 0;
    for (var j = 0; j < res.length; j++) { a += res[j]; if (a >= 65521) a -= 65521; b2 += a; if (b2 >= 65521) b2 -= 65521; }
    var want = ((src[end] << 24) | (src[end + 1] << 16) | (src[end + 2] << 8) | src[end + 3]) >>> 0;
    if ((((b2 << 16) | a) >>> 0) !== want) throw new Error("damaged compressed data (checksum)");
    return res;
  }
  function varints(bytes, n, delta) {  // LEB128 unsigned integers, optionally running sums
    var out = new Uint32Array(n), p = 0, acc = 0;
    for (var i = 0; i < n; i++) {
      var v = 0, sh = 0, b;
      do { b = bytes[p++]; v += (b & 127) * Math.pow(2, sh); sh += 7; } while (b & 128);
      if (delta) { acc += v; out[i] = acc; } else out[i] = v;
    }
    return out;
  }
  var TYPES = { u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array, f8: Float64Array };
  function decodeBlob(b) {
    var bytes = b64bytes(b.d || "");
    if (b.z) bytes = inflateZlib(bytes, b.r || 0);
    if (b.t === "v") return varints(bytes, b.n, !!b.delta);
    if (b.t === "raw") return bytes;
    var T = TYPES[b.t]; if (!T) throw new Error("unknown data type " + b.t);
    if (bytes.byteOffset % T.BYTES_PER_ELEMENT) bytes = bytes.slice();
    var arr = new T(bytes.buffer, bytes.byteOffset, b.n);
    if (b.q) {  // quantised: value = q[0] + code * q[1]; code b.nan = missing
      var f = new Float32Array(b.n), o = b.q[0], s = b.q[1], nan = b.nan === undefined ? -1 : b.nan;
      for (var i = 0; i < b.n; i++) f[i] = arr[i] === nan ? NaN : o + arr[i] * s;
      return f;
    }
    return arr;
  }

  // ------------------------------------------------------------------------------------------------ icons
  var ICON = {
    pan: '<path d="M8 1.5v13M1.5 8h13M8 1.5 6 3.5M8 1.5l2 2M8 14.5l-2-2M8 14.5l2-2M1.5 8l2-2M1.5 8l2 2M14.5 8l-2-2M14.5 8l-2 2"/>',
    zoom: '<circle cx="6.8" cy="6.8" r="4.6"/><path d="m10.2 10.2 4.3 4.3M4.7 6.8h4.2M6.8 4.7v4.2"/>',
    select: '<path stroke-dasharray="2 1.6" d="M2 2.5h12v11H2z"/>',
    lasso: '<path d="M8 2.2c3.6 0 6.3 1.6 6.3 3.6s-2.7 3.6-6.3 3.6S1.7 7.8 1.7 5.8 4.4 2.2 8 2.2Z"/><path d="M4.4 8.8c-.9 1.2-.4 2.6 1 3 1.1.3 1.4 1.3.8 2.4"/>',
    reset: '<path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/>',
    png: '<path d="M8 2v8M4.8 7 8 10.2 11.2 7M2.5 11.5v2h11v-2"/>',
    help: '<circle cx="8" cy="8" r="6.2"/><path d="M6.2 6.3a1.9 1.9 0 1 1 2.6 1.8c-.5.2-.8.6-.8 1.1v.5M8 11.6v.1"/>'
  };
  function icon(name) { return '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + ICON[name] + "</svg>"; }

  // ------------------------------------------------------------------------------------------------ spatial index
  function GridIndex(x, y, n) {
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, i;
    for (i = 0; i < n; i++) { var a = x[i], b = y[i]; if (a === a && b === b) { if (a < x0) x0 = a; if (a > x1) x1 = a; if (b < y0) y0 = b; if (b > y1) y1 = b; } }
    if (!(x1 >= x0)) { x0 = y0 = 0; x1 = y1 = 1; }
    var w = Math.max(x1 - x0, 1e-9), h = Math.max(y1 - y0, 1e-9), cs = Math.sqrt(w * h / Math.max(n / 3, 1));
    var nx = clamp(Math.ceil(w / cs), 1, 2048), ny = clamp(Math.ceil(h / cs), 1, 2048);
    var start = new Int32Array(nx * ny + 1), items = new Int32Array(n), cell = new Int32Array(n);
    for (i = 0; i < n; i++) {
      var cx = Math.floor((x[i] - x0) / w * nx), cy = Math.floor((y[i] - y0) / h * ny);
      if (!(cx >= 0)) { cell[i] = -1; continue; }
      cell[i] = clamp(cy, 0, ny - 1) * nx + clamp(cx, 0, nx - 1); start[cell[i] + 1]++;
    }
    for (i = 0; i < nx * ny; i++) start[i + 1] += start[i];
    var fillp = start.slice(0, nx * ny);
    for (i = 0; i < n; i++) if (cell[i] >= 0) items[fillp[cell[i]]++] = i;
    this.x0 = x0; this.y0 = y0; this.w = w; this.h = h; this.nx = nx; this.ny = ny; this.start = start; this.items = items;
  }
  GridIndex.prototype.nearest = function (x, y, px, py, r, ok) {
    var cx0 = Math.floor((px - r - this.x0) / this.w * this.nx), cx1 = Math.floor((px + r - this.x0) / this.w * this.nx);
    var cy0 = Math.floor((py - r - this.y0) / this.h * this.ny), cy1 = Math.floor((py + r - this.y0) / this.h * this.ny);
    cx0 = clamp(cx0, 0, this.nx - 1); cx1 = clamp(cx1, 0, this.nx - 1); cy0 = clamp(cy0, 0, this.ny - 1); cy1 = clamp(cy1, 0, this.ny - 1);
    var best = -1, bd = r * r;
    for (var cy = cy0; cy <= cy1; cy++) for (var cx = cx0; cx <= cx1; cx++) {
      var c = cy * this.nx + cx;
      for (var k = this.start[c]; k < this.start[c + 1]; k++) {
        var i = this.items[k], dx = x[i] - px, dy = y[i] - py, d = dx * dx + dy * dy;
        if (d <= bd && (!ok || ok(i))) { bd = d; best = i; }
      }
    }
    return best;
  };

  // ------------------------------------------------------------------------------------------------ point stencils
  var STENCILS = {};
  function stencil(r, edge, shape) {
    var key = shape + ":" + r.toFixed(2) + ":" + (edge || 0).toFixed(2);
    if (STENCILS[key]) return STENCILS[key];
    var R = Math.ceil(r + 0.5), fx = [], fy = [], ex = [], ey = [], bx = [], by = [], bw = [], bk = [];
    var rin = Math.max(r - (edge || 0), 0);
    for (var dy = -R; dy <= R; dy++) for (var dx = -R; dx <= R; dx++) {
      var cov = 0, cin = 0;
      for (var sy = 0; sy < 4; sy++) for (var sx = 0; sx < 4; sx++) {
        var ux = dx - 0.375 + sx * 0.25, uy = dy - 0.375 + sy * 0.25, d2 = ux * ux + uy * uy;
        if (d2 <= r * r) { cov++; if (d2 <= rin * rin) cin++; }
      }
      if (!cov) continue;
      cov /= 16; cin /= 16;
      if (!edge) {
        if (cov > 0.999) { fx.push(dx); fy.push(dy); } else { bx.push(dx); by.push(dy); bw.push(cov); bk.push(0); }
      } else if (cov > 0.999 && cin > 0.999) { fx.push(dx); fy.push(dy); }
      else if (cov > 0.999 && cin < 0.001) { ex.push(dx); ey.push(dy); }
      else { bx.push(dx); by.push(dy); bw.push(cov); bk.push(cin / cov); }  // bk: share of the covered part that is fill
    }
    var st = { R: R, fx: new Int32Array(fx), fy: new Int32Array(fy), ex: new Int32Array(ex), ey: new Int32Array(ey),
      bx: new Int32Array(bx), by: new Int32Array(by), bw: new Float32Array(bw), bk: new Float32Array(bk) };
    var keys = Object.keys(STENCILS); if (keys.length > 64) STENCILS = {};
    STENCILS[key] = st;
    return st;
  }
  function blend(d, col, a) {  // col over d with coverage a (both opaque), returns packed
    var b = 1 - a;
    return ((255 << 24) | ((((d >> 16) & 255) * b + ((col >> 16) & 255) * a) << 16) | ((((d >> 8) & 255) * b + ((col >> 8) & 255) * a) << 8) |
      ((d & 255) * b + (col & 255) * a)) >>> 0;
  }

  // ------------------------------------------------------------------------------------------------ the figure
  function Figure(root, spec, opts) {
    this.root = root; this.spec = spec; this.opts = opts || {};
    this.mode = root.getAttribute("data-spc-mode") || "embed";
    this.dark = spec.style !== "light";
    this.cache = {}; this.tables = spec.tables || {}; this.geneCache = {};
    this.state = {}; this.hidden = {}; this.layerOff = {}; this.selection = null; this.highlight = null;
    this.tool = "pan"; this.active = this.mode === "page"; this.renderMs = {};
    var self = this;
    (spec.controls || []).forEach(function (c) { self.state[c.id] = c.value === undefined ? null : c.value; });
    (spec.hidden || []).forEach(function (h) { (self.hidden[h.layer] = self.hidden[h.layer] || {})[h.cat] = true; });
    (spec.layers_off || []).forEach(function (id) { self.layerOff[id] = true; });
    this.build();
  }
  Figure.prototype.blob = function (id) {
    if (id === null || id === undefined) return null;
    if (typeof id !== "string") return id;
    if (this.cache[id]) return this.cache[id];
    var b = (this.spec.blobs || {})[id];
    if (!b) throw new Error("missing data block " + id);
    var v = decodeBlob(b); this.cache[id] = v; return v;
  };
  Figure.prototype.table = function (id) { return typeof id === "string" ? (this.tables[id] || []) : (id || []); };

  Figure.prototype.build = function () {
    var s = this.spec, self = this, root = this.root;
    root.classList.add("spc-root", this.dark ? "spc-dark" : "spc-light");
    root.setAttribute("role", "figure");
    root.setAttribute("aria-label", s.title || "spaclone figure");
    var fb = root.querySelector(".spc-fallback"); if (fb) fb.style.display = "none";
    var head = el("div", "spc-head", root);
    var titles = el("div", "spc-titles", head);
    el("div", "spc-title", titles, s.title || "");
    if (s.subtitle) el("div", "spc-subtitle", titles, s.subtitle);
    this.ctrlBox = el("div", "spc-ctrls", head);
    (s.controls || []).forEach(function (c) { self.addControl(c); });
    this.addTools(head);
    var body = el("div", "spc-body", root);
    this.panelBox = el("div", "spc-panels", body);
    this.side = el("div", "spc-side", body);
    if (s.layout && s.layout.side === "none") this.side.style.display = "none";
    this.legendBox = el("div", "spc-legendbox", this.side);
    this.statsBox = el("div", "spc-stats", this.side);
    var foot = el("div", "spc-foot", root);
    this.notesEl = el("div", "spc-notes", foot);
    var stamp = el("div", "spc-stamp", foot, "spaclone " + (s.version || VERSION));
    if (s.call) stamp.title = s.call;
    this.tip = el("div", "spc-tip", root);
    var lay = s.layout || {};
    this.cols = lay.cols || 1; this.rows = lay.rows || 1;
    this.panelBox.style.gridTemplateColumns = lay.colw ? lay.colw.map(function (w) { return "minmax(0, " + w + "fr)"; }).join(" ") : "repeat(" + this.cols + ", minmax(0, 1fr))";
    this.views = {};
    this.panels = (s.panels || []).map(function (p) { return new Panel(self, p); });
    this.placePanels();
    this.updateControls();
    this.layout();
    this.updateNotes(); this.updateLegend(); this.updateStats();
    if (typeof ResizeObserver !== "undefined") {
      var pending = false;
      this.ro = new ResizeObserver(function () {
        if (pending) return; pending = true;
        global.requestAnimationFrame(function () { pending = false; self.layout(); });
      });
      this.ro.observe(root);
    } else global.addEventListener("resize", function () { self.layout(); });
    if (this.mode !== "page") {
      root.addEventListener("pointerdown", function () { self.active = true; });
      DOC.addEventListener("pointerdown", function (e) { if (!root.contains(e.target)) self.active = false; });
    }
  };

  Figure.prototype.placePanels = function () {  // panels with a "when" are shown only in the states it matches
    var self = this, lay = this.spec.layout || {}, vis = 0, changed = false;
    this.panels.forEach(function (p) {
      var on = matches(p.spec.when, self.state);
      if (p.hidden === on || p.hidden === undefined) changed = true;
      p.hidden = !on; p.el.style.display = on ? "" : "none"; if (on) vis++;
    });
    if (lay.reflow) {  // the shown panels share the row, as many columns as panels
      var k = 0;
      this.panels.forEach(function (p) { if (!p.hidden) { k++; p.el.style.gridColumn = k + " / span 1"; } });
      this.panelBox.style.gridTemplateColumns = "repeat(" + Math.max(vis, 1) + ", minmax(0, 1fr))";
    }
    this.visCols = lay.reflow ? Math.max(vis, 1) : this.cols;
    return changed;
  };
  Figure.prototype.layout = function () {
    var w = this.root.clientWidth || 800, narrow = w < 760, s = this.spec, lay = s.layout || {};
    this.root.classList.toggle("spc-narrow", narrow);
    var sideW = (narrow || this.side.style.display === "none") ? 0 : 250, avail = Math.max(w - 28 - sideW, 200);
    var aspect = (lay.aspect || 1.5) * (lay.reflow ? this.visCols / this.cols : 1), h, cols = this.visCols || this.cols;
    if (this.mode === "page") {
      var head = this.root.querySelector(".spc-head"), foot = this.root.querySelector(".spc-foot");
      var used = (head ? head.offsetHeight : 50) + (foot ? foot.offsetHeight : 24) + 14;
      h = Math.max((global.innerHeight || 800) - used - (narrow ? 160 : 0), 320);
      h = Math.min(h, Math.max(avail / aspect * (cols > 1 ? 1.45 : 1.15), 320));
    } else {
      h = s.height ? s.height : avail / aspect;
      h = clamp(h, lay.min_height || 260, lay.max_height || 900);
    }
    var rowH = (h - (this.rows - 1) * 10) / this.rows;
    this.panelBox.style.gridTemplateRows = "repeat(" + this.rows + ", " + Math.floor(rowH) + "px)";
    this.side.style.maxHeight = "";
    var self = this;
    this.panels.forEach(function (p) { if (!p.hidden) p.resize(); });
    this.panels.forEach(function (p) { if (!p.hidden) p.fitIfNeeded(); });
    this.renderAll();
    this.notifyHeight();
    void self;
  };
  Figure.prototype.notifyHeight = function () {  // a figure inside an iframe tells the page around it how tall it is
    if (this.mode !== "embed" || global.parent === global) return;
    var h = this.root.offsetHeight;
    try { var fe = global.frameElement; if (fe && fe.style.opacity !== "0") fe.style.height = h + "px"; } catch (e) { /* cross-origin */ }
    try { global.parent.postMessage({ spaclone: "height", id: this.root.id, height: h }, "*"); } catch (e2) { /* no parent */ }
  };
  Figure.prototype.renderAll = function () { this.panels.forEach(function (p) { p.render(); }); };
  Figure.prototype.overlays = function () { this.panels.forEach(function (p) { p.drawOverlay(); }); };
  Figure.prototype.renderLinked = function (panel) {
    var g = panel.spec.link;
    this.panels.forEach(function (p) { if (p === panel || (g && p.spec.link === g)) p.render(); });
  };
  Figure.prototype.view = function (panel) {
    var g = panel.spec.link;
    if (!g) return panel.ownView || (panel.ownView = {});
    return this.views[g] || (this.views[g] = {});
  };

  // ---- controls
  Figure.prototype.addControl = function (c) {
    var self = this, box = el("div", "spc-ctrl", this.ctrlBox);
    box.setAttribute("data-control", c.id);
    if (c.label) el("span", "spc-lab", box, c.label);
    if (c.kind === "select") {
      var sel = el("select", "spc-select", box);
      sel.setAttribute("aria-label", c.label || c.id);
      (c.options || []).forEach(function (o) {
        var op = el("option", null, sel, o.label === undefined ? o.value : o.label);
        op.value = String(o.value);
      });
      sel.value = String(this.state[c.id]);
      sel.addEventListener("change", function () {
        var o = (c.options || []).filter(function (x) { return String(x.value) === sel.value; })[0];
        self.setState(c.id, o ? o.value : sel.value);
      });
      this["ctl_" + c.id] = { set: function (v) { sel.value = String(v); } };
    } else if (c.kind === "search") {
      this.addSearch(c, box);
    }
  };
  Figure.prototype.searchOptions = function (c) {
    if (c.source && (c.source.genes_by || c.source.genes)) {
      var g = this.genes(c.source.genes_by ? this.state[c.source.genes_by] : c.source.genes);
      return g ? g.names : [];
    }
    if (c.source && c.source.by) {
      var key = this.state[c.source.by];
      return this.table((c.source.tables || {})[key]);
    }
    return this.table(c.source);
  };
  Figure.prototype.searchAux = function (c) {
    if (c.source && c.source.by && c.source.aux) return this.table(c.source.aux[this.state[c.source.by]]);
    if (c.aux) return this.table(c.aux);
    return null;
  };
  Figure.prototype.addSearch = function (c, box) {
    var self = this, wrap = el("div", "spc-search", box);
    var inp = el("input", "spc-input", wrap);
    inp.type = "text"; inp.spellcheck = false; inp.autocomplete = "off";
    inp.placeholder = c.placeholder || "search";
    inp.setAttribute("aria-label", c.label || c.id);
    inp.setAttribute("role", "combobox"); inp.setAttribute("aria-expanded", "false"); inp.setAttribute("aria-autocomplete", "list");
    var list = el("ul", "spc-list", wrap); list.setAttribute("role", "listbox"); list.style.display = "none";
    var msg = el("span", "spc-msgline", box);
    var chips = c.multi ? el("div", "spc-chips", box) : null;
    var current = [], active = -1;
    function names() { return self.searchOptions(c); }
    function find(q) {
      var all = names(), aux = self.searchAux(c), ql = q.trim().toLowerCase(), pre = [], sub = [];
      if (!ql) { for (var i = 0; i < Math.min(all.length, 40); i++) pre.push(i); return pre; }
      for (var j = 0; j < all.length; j++) {
        var nm = String(all[j]).toLowerCase(), ix = nm.indexOf(ql);
        if (ix === 0) pre.push(j); else if (ix > 0) sub.push(j);
        if (pre.length >= 60) break;
      }
      void aux;
      return pre.concat(sub).slice(0, 60);
    }
    function show(q) {
      current = find(q); active = current.length ? 0 : -1;
      list.innerHTML = "";
      var all = names(), aux = self.searchAux(c);
      if (!current.length) {
        var li = el("li", "spc-none", list, q.trim() ? "no " + (c.noun || "match") + " named \u201c" + q.trim() + "\u201d in this file (" + fmtInt(all.length) + " " + (c.nouns || "names") + " embedded)" : "nothing to choose");
        li.setAttribute("role", "option");
      }
      current.forEach(function (ix, k) {
        var li = el("li", null, list); li.setAttribute("role", "option"); li.setAttribute("aria-selected", k === active ? "true" : "false");
        el("span", null, li, String(all[ix]));
        if (aux) el("span", "spc-aux", li, String(aux[ix]));
        li.addEventListener("mousedown", function (e) { e.preventDefault(); pick(ix); });
      });
      list.style.display = "block"; inp.setAttribute("aria-expanded", "true");
    }
    function hide() { list.style.display = "none"; inp.setAttribute("aria-expanded", "false"); }
    function pick(ix) {
      var all = names(), v = String(all[ix]);
      inp.classList.remove("spc-bad"); msg.textContent = ""; hide();
      if (c.multi) {
        var cur = (self.state[c.id] || []).slice();
        if (cur.indexOf(v) < 0) { cur.push(v); if (cur.length > (c.max || 8)) cur.shift(); }
        inp.value = ""; self.setState(c.id, cur); renderChips();
      } else { inp.value = v; self.setState(c.id, v); }
    }
    function submit() {
      var q = inp.value.trim(); if (!q) { if (!c.multi) { self.setState(c.id, null); } hide(); return; }
      var all = names(), ql = q.toLowerCase(), exact = -1;
      for (var j = 0; j < all.length; j++) if (String(all[j]).toLowerCase() === ql) { exact = j; break; }
      if (exact >= 0) return pick(exact);
      if (active >= 0 && current.length) return pick(current[active]);
      inp.classList.add("spc-bad");
      msg.textContent = "no " + (c.noun || "match") + " \u201c" + q + "\u201d";
    }
    function renderChips() {
      if (!chips) return; chips.innerHTML = "";
      var cols = c.colors || [];
      (self.state[c.id] || []).forEach(function (v, k) {
        var ch = el("span", "spc-chip", chips), dot = el("i", null, ch);
        dot.style.background = cols[k] || "#888";
        el("span", null, ch, v);
        var x = el("button", null, ch, "\u00d7"); x.title = "remove"; x.setAttribute("aria-label", "remove " + v);
        x.addEventListener("click", function () {
          var cur = (self.state[c.id] || []).filter(function (w) { return w !== v; });
          self.setState(c.id, cur); renderChips();
        });
      });
    }
    inp.addEventListener("input", function () { inp.classList.remove("spc-bad"); msg.textContent = ""; show(inp.value); });
    inp.addEventListener("focus", function () { show(inp.value); });
    inp.addEventListener("blur", function () { global.setTimeout(hide, 120); });
    inp.addEventListener("keydown", function (e) {
      var items = list.querySelectorAll("li[role=option]");
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault(); if (!current.length) return;
        active = (active + (e.key === "ArrowDown" ? 1 : -1) + current.length) % current.length;
        for (var k = 0; k < items.length; k++) items[k].setAttribute("aria-selected", k === active ? "true" : "false");
        if (items[active]) items[active].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") { e.preventDefault(); submit(); }
      else if (e.key === "Escape") { hide(); inp.blur(); }
    });
    if (!c.multi && this.state[c.id]) inp.value = this.state[c.id];
    renderChips();
    this["ctl_" + c.id] = { set: function (v) { if (c.multi) renderChips(); else inp.value = v || ""; }, input: inp, msg: msg, submit: submit };
  };
  Figure.prototype.setState = function (id, value) {
    this.state[id] = value;
    var ui = this["ctl_" + id]; if (ui) ui.set(value);
    var ctl0 = (this.spec.controls || []).filter(function (c) { return c.id === id; })[0], me = this;
    if (ctl0 && ctl0.also && value !== null && value !== undefined && !(Array.isArray(value) && !value.length)) {
      Object.keys(ctl0.also).forEach(function (k) { me.state[k] = ctl0.also[k]; var u = me["ctl_" + k]; if (u) u.set(ctl0.also[k]); });
    }
    var dep = (this.spec.controls || []).filter(function (c) { return c.reset_on && c.reset_on.indexOf(id) >= 0; });
    var self = this;
    dep.forEach(function (c) { self.state[c.id] = c.multi ? [] : null; var ctl = self["ctl_" + c.id]; if (ctl) ctl.set(self.state[c.id]); });
    if (this.selection && this.selection.panel) {
      var p = this.panels.filter(function (q) { return q.spec.id === self.selection.panel; })[0];
      if (p && p.spec.home_by && p.spec.home_by === id) this.selection = null;
    }
    var moved = this.placePanels();
    this.panels.forEach(function (p) { p.stateChanged(id); });
    this.updateControls(); this.updateNotes(); this.updateLegend(); this.updateStats();
    if (moved) this.layout(); else this.renderAll();
  };
  Figure.prototype.updateControls = function () {
    var self = this;
    (this.spec.controls || []).forEach(function (c) {
      var box = self.ctrlBox.querySelector('[data-control="' + c.id + '"]');
      if (box && c.when) { var on = matches(c.when, self.state); box.style.visibility = on ? "" : "hidden"; box.setAttribute("aria-hidden", on ? "false" : "true"); }
    });
  };
  Figure.prototype.addTools = function (head) {
    var self = this, s = this.spec, tools = (s.layout && s.layout.tools) || ["pan", "zoom", "select", "lasso", "reset", "png"];
    if (!tools.length) return;
    var box = el("div", "spc-tools", head);
    box.setAttribute("role", "toolbar"); box.setAttribute("aria-label", "figure tools");
    var names = { pan: "Pan (drag); scroll or pinch to zoom", zoom: "Zoom to a box (drag)", select: "Select a box: counts in the side panel",
      lasso: "Select a free shape: counts in the side panel", reset: "Reset the view (double-click)", png: "Save the current view as PNG",
      help: "Keys: + / \u2212 zoom, arrows pan, 0 reset, Esc clear" };
    this.toolButtons = {};
    tools.forEach(function (t) {
      if (t === "|") { el("span", "spc-tool-sep", box); return; }
      var b = el("button", "spc-tool", box); b.type = "button"; b.innerHTML = icon(t); b.title = names[t] || t;
      b.setAttribute("aria-label", names[t] || t); b.setAttribute("data-tool", t);
      if (t === "pan" || t === "zoom" || t === "select" || t === "lasso") {
        b.setAttribute("aria-pressed", t === self.tool ? "true" : "false");
        b.addEventListener("click", function () { self.setTool(t); });
      } else if (t === "reset") b.addEventListener("click", function () { self.resetViews(); });
      else if (t === "png") b.addEventListener("click", function () { self.savePng(); });
      self.toolButtons[t] = b;
    });
  };
  Figure.prototype.setTool = function (t) {
    this.tool = t;
    for (var k in this.toolButtons) if (this.toolButtons[k].hasAttribute("aria-pressed")) this.toolButtons[k].setAttribute("aria-pressed", k === t ? "true" : "false");
    this.panels.forEach(function (p) { p.el.className = p.el.className.replace(/ ?spc-mode-\w+/g, "") + " spc-mode-" + t; });
  };
  Figure.prototype.resetViews = function () { this.views = {}; this.panels.forEach(function (p) { p.ownView = null; p.home(); }); this.renderAll(); };
  Figure.prototype.savePng = function () {
    var dpr = global.devicePixelRatio || 1, box = this.panelBox.getBoundingClientRect();
    var cv = DOC.createElement("canvas"); cv.width = Math.round(box.width * dpr); cv.height = Math.round(box.height * dpr);
    var ctx = cv.getContext("2d"); ctx.fillStyle = this.dark ? "#000000" : "#ffffff"; ctx.fillRect(0, 0, cv.width, cv.height);
    this.panels.forEach(function (p) {
      if (p.hidden) return;
      var r = p.el.getBoundingClientRect(); ctx.drawImage(p.canvas, Math.round((r.left - box.left) * dpr), Math.round((r.top - box.top) * dpr));
    });
    var name = (this.spec.file_stem || "spaclone_figure") + ".png", a = DOC.createElement("a");
    try {
      a.href = cv.toDataURL("image/png"); a.download = name; DOC.body.appendChild(a); a.click(); a.remove();
    } catch (e) { this.flash("could not save the PNG here: " + e.message); }
  };
  Figure.prototype.flash = function (text) {
    var p = this.panels[0]; if (!p) return;
    p.hint.textContent = text; p.hint.classList.add("spc-on");
    global.clearTimeout(p._hintT); p._hintT = global.setTimeout(function () { p.hint.classList.remove("spc-on"); }, 1800);
  };

  // ---- notes, legend, selection statistics
  Figure.prototype.updateNotes = function () {
    var self = this; this.notesEl.innerHTML = "";
    (this.spec.notes || []).forEach(function (n) {
      var t = typeof n === "string" ? { text: n } : n;
      if (!matches(t.when, self.state)) return;
      el("div", t.warn ? "spc-warnnote" : null, self.notesEl, t.text);
    });
  };
  Figure.prototype.legendSections = function () {
    var self = this;
    var secs = (this.spec.legend || []).filter(function (s) { return matches(s.when, self.state); });
    function lines(s) { return !s.kind && (s.items || []).length && (s.items || []).every(function (it) { return it.shape === "line" || it.shape === "dash" || it.shape === "box"; }); }
    return secs.filter(function (s) { return !lines(s); }).concat(secs.filter(lines));
  };
  Figure.prototype.updateLegend = function () {
    var self = this, box = this.legendBox; box.innerHTML = "";
    this.legendSections().forEach(function (sec) {
      var wrap = el("div", "spc-legsec", box);
      if (sec.kind === "colorbar" || sec.kind === "genebar") { self.colorbar(wrap, sec); return; }
      if (sec.kind === "highlight") {
        if (sec.title) el("div", "spc-sec-title", wrap, sec.title);
        var chosen = self.state[sec.control] || [], lst = el("div", "spc-legend", wrap);
        if (!chosen.length) { el("div", "spc-empty", wrap, sec.empty || "search above"); return; }
        chosen.forEach(function (v, k) {
          var row = el("div", "spc-li", lst), sw = el("span", "spc-sw", row);
          sw.style.background = (sec.colors || [])[Math.min(k, (sec.colors || []).length - 1)] || "#888";
          el("span", null, row, v);
        });
        if (sec.other) { var o = el("div", "spc-li", lst), so = el("span", "spc-sw", o); so.style.background = sec.other_color || "#555"; el("span", null, o, sec.other); }
        return;
      }
      if (sec.title) el("div", "spc-sec-title", wrap, sec.title);
      var list = el("div", "spc-legend", wrap);
      var counts = sec.counts_by ? (sec.counts_by[self.state[sec.counts_key]] || null) : null;
      (sec.items || []).forEach(function (it, k) {
        var tog = !!(it.layer && (sec.toggle !== false));
        var b = el(tog ? "button" : "div", "spc-li" + (tog ? " spc-toggle" : ""), list);
        if (tog) b.type = "button";
        var sw = el("span", "spc-sw" + (it.shape && it.shape !== "point" ? " spc-" + it.shape : ""), b);
        if (it.shape === "line" || it.shape === "dash" || it.shape === "box") sw.style.borderColor = it.color; else sw.style.background = it.color;
        el("span", null, b, it.label);
        var n = counts ? counts[k] : it.n;
        if (n !== undefined && n !== null) el("span", "spc-n", b, fmtInt(n));
        var off = it.cat !== undefined ? !!(self.hidden[it.layer] && self.hidden[it.layer][it.cat]) : !!self.layerOff[it.layer];
        if (off) b.classList.add("spc-off");
        if (tog) {
          b.setAttribute("aria-pressed", off ? "false" : "true");
          b.title = "click: show / hide; double-click: show only this";
          b.addEventListener("click", function () { self.toggleItem(it); });
          b.addEventListener("dblclick", function (e) { e.preventDefault(); self.soloItem(sec, it); });
        }
      });
    });
  };
  Figure.prototype.toggleItem = function (it) {
    if (it.cat !== undefined) { var h = (this.hidden[it.layer] = this.hidden[it.layer] || {}); h[it.cat] = !h[it.cat]; }
    else this.layerOff[it.layer] = !this.layerOff[it.layer];
    this.panels.forEach(function (p) { p.invalidate(it.layer); });
    this.updateLegend(); this.updateStats(); this.renderAll();
  };
  Figure.prototype.soloItem = function (sec, it) {
    var self = this;
    (sec.items || []).forEach(function (o) {
      if (o.cat === undefined || o.layer !== it.layer) return;
      (self.hidden[o.layer] = self.hidden[o.layer] || {})[o.cat] = o.cat !== it.cat;
    });
    this.panels.forEach(function (p) { p.invalidate(it.layer); });
    this.updateLegend(); this.updateStats(); this.renderAll();
  };
  Figure.prototype.colorbar = function (wrap, sec) {
    var info = sec;
    if (sec.kind === "genebar") {
      info = this.geneBar(sec);
      if (!info) { el("div", "spc-empty", wrap, sec.empty || "pick a gene"); return; }
    }
    var cb = el("div", "spc-cbar", wrap);
    el("div", "spc-cb-lab", cb, info.label || "");
    var ramp = el("div", "spc-cb-ramp", cb); ramp.style.background = rampCss(info.ramp);
    var ticks = el("div", "spc-cb-ticks", cb), lo = info.domain[0], hi = info.domain[1], f = info.fmt;
    var vals = info.ticks || [lo, hi];
    vals.forEach(function (v) {
      var sp = el("span", null, ticks, info.ticklabels ? info.ticklabels[vals.indexOf(v)] : fmtNum(v, f));
      var t = info.scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / ((hi - lo) || 1);
      sp.style.left = (100 * clamp(t, 0, 1)) + "%";
    });
    if (info.note) el("div", "spc-cb-note", cb, info.note);
    cb.setAttribute("data-domain", lo + "," + hi);
    cb.setAttribute("data-label", info.label || "");
  };
  Figure.prototype.updateStats = function () {
    var box = this.statsBox, sel = this.selection, self = this;
    box.innerHTML = "";
    if (!this.spec.stats || !Object.keys(this.spec.stats).length) return;
    el("div", "spc-sec-title", box, "selection");
    if (!sel) {
      el("div", "spc-empty", box, "Draw a box or a free shape (tools above) to count what is inside.");
      return;
    }
    var res = this.computeStats(sel);
    this.lastStats = res;
    if (!res.length) { el("div", "spc-empty", box, "nothing selectable is shown"); return; }
    function srow(k, v, cls) { var r = el("div", "spc-srow" + (cls ? " " + cls : ""), box); el("span", "spc-k", r, k); el("span", "spc-v", r, v); r.title = k + ": " + v; return r; }
    res.forEach(function (r) {
      srow(r.unit, fmtInt(r.n), "spc-strong");
      r.rows.forEach(function (q) {
        if (q.kind === "shares") {
          if (!q.total) { srow(q.label, "none"); return; }
          el("div", "spc-shead", box, q.label + (q.total < r.n ? " \u00b7 " + fmtInt(q.total) + " with one" : ""));
          var bar = el("div", "spc-bar", box);
          q.items.forEach(function (it) { if (it.share > 0) { var i = el("i", null, bar); i.style.width = (100 * it.share) + "%"; i.style.background = it.color; i.title = it.name + " " + fmtNum(it.share, "pct"); } });
          var shown = q.items.filter(function (it) { return it.count > 0; }), lim = q.show || 6;
          shown.slice(0, lim).forEach(function (it) { srow(it.name, fmtNum(it.share, "pct") + " \u00b7 " + fmtInt(it.count), "spc-sub"); });
          if (shown.length > lim) {
            var rest = shown.slice(lim).reduce(function (a, it) { return a + it.count; }, 0);
            srow((shown.length - lim) + " more", fmtNum(rest / q.total, "pct") + " \u00b7 " + fmtInt(rest), "spc-sub");
          }
        } else srow(q.label, q.kind === "distinct" || q.kind === "count" ? fmtInt(q.value) : fmtNum(q.value, q.fmt));
      });
    });
    var btns = el("div", "spc-btnrow", box);
    var clear = el("button", "spc-btn", btns, "clear"); clear.type = "button";
    clear.addEventListener("click", function () { self.selection = null; self.updateStats(); self.overlays(); });
    var copy = el("button", "spc-btn", btns, "copy shape (\u00b5m)"); copy.type = "button";
    copy.title = "Copy the shape's corners in the section's original micrometre coordinates (JSON), e.g. for matplotlib.path.Path(...).contains_points";
    copy.addEventListener("click", function () { self.copyShape(); });
  };
  Figure.prototype.shapeUm = function () {
    var sel = this.selection; if (!sel) return null;
    var p = this.panels.filter(function (q) { return q.spec.id === sel.panel; })[0];
    var fr = (this.spec.frames || {})[p && p.spec.home_by ? this.state[p.spec.home_by] : "_"] || (this.spec.frames || {})._;
    return sel.xs.map(function (x, i) {
      var y = sel.ys[i];
      if (!fr) return [x, y];
      return [fr[0] * x + fr[2] * y + fr[4], fr[1] * x + fr[3] * y + fr[5]];
    });
  };
  Figure.prototype.copyShape = function () {
    var pts = this.shapeUm(); if (!pts) return;
    var txt = JSON.stringify(pts.map(function (p) { return [+p[0].toFixed(2), +p[1].toFixed(2)]; })), self = this;
    function fallback() {
      var ta = el("textarea", null, DOC.body); ta.value = txt; ta.select();
      try { DOC.execCommand("copy"); self.flash("shape copied"); } catch (e) { self.flash("copy not allowed here"); }
      ta.remove();
    }
    if (global.navigator && global.navigator.clipboard && global.navigator.clipboard.writeText) {
      global.navigator.clipboard.writeText(txt).then(function () { self.flash("shape copied"); }, fallback);
    } else fallback();
  };
  Figure.prototype.computeStats = function (sel) {
    var self = this, out = [];
    var p = this.panels.filter(function (q) { return q.spec.id === sel.panel; })[0];
    if (!p) return out;
    p.visibleLayers().forEach(function (L) {
      if (L.type !== "points" || !L.select) return;
      var st = self.spec.stats[L.select]; if (!st) return;
      var x = self.blob(L.x), y = self.blob(L.y), n = L.n, inside = [];
      var bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity, k;
      for (k = 0; k < sel.xs.length; k++) { bx0 = Math.min(bx0, sel.xs[k]); bx1 = Math.max(bx1, sel.xs[k]); by0 = Math.min(by0, sel.ys[k]); by1 = Math.max(by1, sel.ys[k]); }
      for (var i = 0; i < n; i++) {
        var a = x[i], b = y[i];
        if (!(a >= bx0 && a <= bx1 && b >= by0 && b <= by1)) continue;
        if (sel.box || pip(a, b, sel.xs, sel.ys)) inside.push(i);
      }
      var res = { layer: L.id, unit: st.unit || "points", n: inside.length, rows: [] };
      (st.rows || []).forEach(function (q) {
        if (q.kind === "distinct") {
          var codes = self.blob(q.codes), seen = new Set(), skip = q.skip === undefined ? -1 : q.skip;
          inside.forEach(function (i) { var c = codes[i]; if (c !== skip) seen.add(c); });
          res.rows.push({ kind: "distinct", label: q.label, value: seen.size });
        } else if (q.kind === "shares") {
          var cc = self.blob(q.codes), cnt = new Float64Array(q.names.length), tot = 0, skip2 = q.skip || [];
          inside.forEach(function (i) { var c = cc[i]; if (c < q.names.length && skip2.indexOf(c) < 0) { cnt[c]++; tot++; } });
          var items = q.names.map(function (nm, j) { return { name: nm, color: (q.colors || [])[j] || "#888", count: cnt[j], share: tot ? cnt[j] / tot : 0 }; });
          items.sort(function (u, v) { return v.count - u.count; });
          res.rows.push({ kind: "shares", label: q.label, items: items, total: tot, show: q.show });
        } else if (q.kind === "mean" || q.kind === "sum" || q.kind === "frac") {
          var vals = self.blob(q.values), s = 0, m = 0;
          inside.forEach(function (i) { var v = vals[i]; if (v === v) { s += v; m++; } });
          res.rows.push({ kind: q.kind, label: q.label, fmt: q.fmt, value: q.kind === "sum" ? s : (m ? s / m : NaN), n: m });
        }
      });
      out.push(res);
    });
    return out;
  };

  // ---- genes
  Figure.prototype.genes = function (key) {
    var g = (this.spec.genes || {})[key];
    if (!g) return null;
    if (!g._names) {
      g._names = this.table(g.names);
      g._lower = {};
      for (var i = 0; i < g._names.length; i++) { var nm = String(g._names[i]).toLowerCase(); if (!(nm in g._lower)) g._lower[nm] = i; }
    }
    return { names: g._names, spec: g, lower: g._lower };
  };
  Figure.prototype.geneValues = function (key, name) {
    var ck = key + "\u0000" + name + "\u0000" + (this.state._norm || "");
    if (this.geneCache[ck]) return this.geneCache[ck];
    var G = this.genes(key); if (!G) return null;
    var gi = G.lower[String(name).toLowerCase()]; if (gi === undefined) return null;
    var g = G.spec, ptr = this.blob(g.ptr), bptr = this.blob(g.bptr), idxBytes = this.blob(g.idx), vals = this.blob(g.val);
    var a = ptr[gi], b = ptr[gi + 1], nnz = b - a;
    var ix = varints(idxBytes.subarray(bptr[gi], bptr[gi + 1]), nnz, true);
    var out = new Float32Array(g.n), scale = g.scale ? this.blob(g.scale)[gi] : 1, norm = g.counts && g.norm !== false && this.normMode(g) === "norm";
    var tot = norm && g.totals ? this.blob(g.totals) : null, mx = 0, nz = [];
    for (var k = 0; k < nnz; k++) {
      var v = vals[a + k] * scale;
      if (tot) v = Math.log1p(v / (tot[ix[k]] || 1) * (g.target || 1e4));
      out[ix[k]] = v; if (v > mx) mx = v; nz.push(v);
    }
    nz.sort(function (u, w) { return u - w; });
    var hi = nz.length ? nz[Math.min(nz.length - 1, Math.floor(0.99 * (nz.length - 1)))] : 0;
    if (!(hi > 0)) hi = mx > 0 ? mx : 1;
    var res = { values: out, nnz: nnz, max: mx, hi: hi, name: G.names[gi], unit: g.counts ? (norm ? (g.norm_label || "log1p(counts per 10k)") : "counts") : (g.unit || "value"), step: g.counts ? 0 : scale };
    var keys = Object.keys(this.geneCache); if (keys.length > 24) this.geneCache = {};
    this.geneCache[ck] = res;
    return res;
  };
  Figure.prototype.normMode = function (g) { return this.state._norm || g.default_norm || "norm"; };
  Figure.prototype.geneBar = function (sec) {
    var key = sec.genes, name = this.state[sec.gene];
    if (!name) return null;
    var gv = this.geneValues(key, name); if (!gv) return null;
    var note = fmtInt(gv.nnz) + " " + (sec.unit_word || "units") + " detect it \u00b7 colour capped at their 99th percentile (max " + fmtNum(gv.max) + ")";
    if (gv.step) note += " \u00b7 stored to \u00b1" + fmtNum(gv.step / 2);
    return { label: gv.name + " \u00b7 " + gv.unit, ramp: sec.ramp, domain: [0, gv.hi], note: note };
  };

  // ------------------------------------------------------------------------------------------------ panels
  function Panel(fig, spec) {
    this.fig = fig; this.spec = spec;
    var el0 = el("div", "spc-panel spc-mode-" + fig.tool, fig.panelBox);
    var a = spec.area || [0, 0, 1, 1];
    el0.style.gridColumn = (a[0] + 1) + " / span " + (a[2] || 1);
    el0.style.gridRow = (a[1] + 1) + " / span " + (a[3] || 1);
    el0.tabIndex = 0;
    el0.setAttribute("aria-label", spec.title || spec.id);
    this.el = el0;
    this.canvas = el("canvas", null, el0);
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", (spec.title || spec.id) + ": drag to pan, scroll to zoom");
    this.over = el("canvas", "spc-over", el0);  // hover ring, linked highlights, selections: redrawn without the data
    this.over.style.pointerEvents = "none";
    this.octx = this.over.getContext("2d");
    this.titleEl = el("div", "spc-ptitle", el0, "");
    this.noteEl = el("div", "spc-pnote", el0, "");
    this.hint = el("div", "spc-hint", el0, "");
    this.cover = el("div", "spc-cover", el0, ""); this.cover.style.display = "none";
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true, alpha: false }) || this.canvas.getContext("2d");
    this.indexes = {}; this.prepared = {}; this.images = {}; this.lastMs = 0; this.w = 0; this.h = 0;
    this.isMap = spec.kind === "map";
    this.bind();
    this.updateTitle();
  }
  Panel.prototype.updateTitle = function () {
    var t = this.spec.title || "", tw = this.spec.title_when || [];
    for (var i = 0; i < tw.length; i++) if (matches(tw[i].when, this.fig.state)) { t = tw[i].title; break; }
    if (this.spec.title_gene && this.fig.state[this.spec.title_gene]) t = t ? t + " \u00b7 " + this.fig.state[this.spec.title_gene] : this.fig.state[this.spec.title_gene];
    var sub = "", sw = this.spec.subtitle_when || [];
    for (var j = 0; j < sw.length; j++) if (matches(sw[j].when, this.fig.state)) { sub = sw[j].text; break; }
    this.titleEl.textContent = "";
    if (t) el("span", "spc-pt", this.titleEl, t);
    if (sub) el("span", "spc-psub", this.titleEl, sub);
    this.titleEl.style.display = (t || sub) ? "" : "none";
  };
  Panel.prototype.resize = function () {
    var r = this.el.getBoundingClientRect(), dpr = global.devicePixelRatio || 1;
    var w = Math.max(Math.round(r.width), 10), h = Math.max(Math.round(r.height), 10);
    var changed = w !== this.w || h !== this.h || dpr !== this.dpr;
    this.w = w; this.h = h; this.dpr = dpr;
    if (changed) {
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
      this.over.width = this.canvas.width; this.over.height = this.canvas.height;
      this.plot = this.plotRect();
      this.fresh = false; this.snapView = null;
    }
    return changed;
  };
  Panel.prototype.visibleLayers = function () {
    var f = this.fig, st = f.state, k = this.isMap ? this.zoomRatio() : 1, self0 = this;
    return (this.spec.layers || []).filter(function (L) {
      if (!matches(L.when, st)) return false;
      if (f.layerOff[L.okey || L.id]) return false;
      if (L.min_zoom && k < L.min_zoom) return false;
      if (L.max_view_um && self0.isMap) { var vv = f.view(self0); if (vv.k && self0.w / vv.k > L.max_view_um) return false; }
      if (L.needs && !st[L.needs]) return false;
      if (L.needs_any && !(st[L.needs_any] && st[L.needs_any].length)) return false;
      return true;
    });
  };
  Panel.prototype.homeBox = function () {
    var s = this.spec, hb = s.home;
    if (s.home_by) hb = (s.home_map || {})[this.fig.state[s.home_by]] || hb;
    return hb || [0, 1, 0, 1];
  };
  Panel.prototype.fitK = function () {
    var b = this.homeBox(), pad = this.spec.pad === undefined ? 0.03 : this.spec.pad, bw = (b[1] - b[0]) || 1, bh = (b[3] - b[2]) || 1;
    var k = Math.min(this.w / (bw * (1 + 2 * pad)), this.h / (bh * (1 + 2 * pad)));
    return k > 0 && isFinite(k) ? k : 1;
  };
  Panel.prototype.home = function () {
    var v = this.fig.view(this), b = this.homeBox(), self = this;
    if (this.isMap) {
      var k = this.fitK();
      if (this.spec.link) this.fig.panels.forEach(function (p) { if (p !== self && p.spec.link === self.spec.link && p.w && !p.hidden) k = Math.min(k, p.fitK()); });
      v.cx = (b[0] + b[1]) / 2; v.cy = (b[2] + b[3]) / 2; v.k = k; v.k0 = k;
    } else {
      var ax = this.spec.axes || {}, xr = this.axisRange(ax.x || {}), yr = this.axisRange(ax.y || {});
      if (ax.y_by) for (var yk in ax.y_by) { var rr = ax.y_by[yk][this.fig.state[yk]]; if (rr) yr = this.axisRange({ range: rr, scale: (ax.y || {}).scale }); }
      if (ax.x_by) for (var xk in ax.x_by) { var rx = ax.x_by[xk][this.fig.state[xk]]; if (rx) xr = this.axisRange({ range: rx, scale: (ax.x || {}).scale }); }
      v.x0 = xr[0]; v.x1 = xr[1]; v.y0 = yr[0]; v.y1 = yr[1];
    }
    v.homeKey = this.homeKey(); v.atHome = true;
  };
  Panel.prototype.homeKey = function () { return this.spec.home_by ? String(this.fig.state[this.spec.home_by]) : "_"; };
  Panel.prototype.fitIfNeeded = function () {
    var v = this.fig.view(this);
    if (this.isMap ? !(v.k > 0) : v.x0 === undefined) this.home();
    else if (v.homeKey !== this.homeKey() || v.atHome) this.home();
  };
  Panel.prototype.stateChanged = function (id) {
    if (this.spec.home_by === id) { var v = this.fig.view(this); for (var k in v) delete v[k]; this.home(); }
    this.updateTitle();
    if (this.spec.reset_on && this.spec.reset_on.indexOf(id) >= 0) this.home();
    if (this.spec.axes && ((this.spec.axes.y_by && this.spec.axes.y_by[id]) || (this.spec.axes.x_by && this.spec.axes.x_by[id]))) this.home();
  };
  Panel.prototype.invalidate = function (key) {
    var self = this;
    (this.spec.layers || []).forEach(function (L) { if (L.id === key || L.hkey === key) delete self.prepared[L.id]; });
  };
  Panel.prototype.zoomRatio = function () { var v = this.fig.view(this); return v.k0 ? v.k / v.k0 : 1; };
  Panel.prototype.axisRange = function (a) {
    if (a.range) return a.scale === "log" ? [Math.log10(a.range[0]), Math.log10(a.range[1])] : a.range.slice();
    if (a.scale === "cat") return [-0.5, (a.cats || []).length - 0.5];
    return [0, 1];
  };
  Panel.prototype.plotRect = function () {
    if (this.isMap) return { l: 0, t: 0, r: this.w, b: this.h };
    var ax = this.spec.axes || {}, m = this.spec.margin || {};
    var l = m.l !== undefined ? m.l : (ax.y && ax.y.hide ? 12 : 58), b = m.b !== undefined ? m.b : (ax.x && ax.x.hide ? 10 : 42);
    if (ax.y && ax.y.scale === "cat" && !ax.y.hide) {
      var longest = 0, ctx = this.ctx; ctx.font = "11.5px " + FONT;
      (ax.y.cats || []).forEach(function (c) { longest = Math.max(longest, ctx.measureText(String(c)).width); });
      l = Math.max(l, Math.min(longest + 22 + (ax.y.label ? 14 : 0), this.w * 0.42));  // room for the axis title
    }
    if (ax.x && ax.x.scale === "cat" && !ax.x.hide && ax.x.rotate) {
      var lx = 0, c2 = this.ctx; c2.font = "11.5px " + FONT;
      (ax.x.cats || []).forEach(function (c) { lx = Math.max(lx, c2.measureText(String(c)).width); });
      b = Math.max(b, Math.min(lx * 0.75 + 30, this.h * 0.45));
    }
    var rect = { l: l, t: m.t !== undefined ? m.t : (this.spec.title ? 30 : 12), r: this.w - (m.r !== undefined ? m.r : 16), b: this.h - b, lm: l, bm: b };
    if (ax.square && ax.x && ax.y) {  // square cells (a matrix): the plot keeps the top-left corner and shrinks
      var nx = Math.max(1, (ax.x.cats || []).length), ny = Math.max(1, (ax.y.cats || []).length);
      var W = rect.r - rect.l, H = rect.b - rect.t;
      if (W > 0 && H > 0) { if (W / H > nx / ny) rect.r = rect.l + H * nx / ny; else rect.b = rect.t + W * ny / nx; }
    }
    return rect;
  };
  // transform data -> CSS px
  Panel.prototype.tf = function () {
    var v = this.fig.view(this);
    if (this.isMap) {
      var inv = this.spec.invert_y !== false, k = v.k, w2 = this.w / 2, h2 = this.h / 2;
      return { sx: k, tx: w2 - v.cx * k, sy: inv ? k : -k, ty: inv ? h2 - v.cy * k : h2 + v.cy * k, xlog: false, ylog: false };
    }
    var p = this.plot, ax = this.spec.axes || {}, xl = ax.x && ax.x.scale === "log", yl = ax.y && ax.y.scale === "log";
    var sx = (p.r - p.l) / ((v.x1 - v.x0) || 1), sy = -(p.b - p.t) / ((v.y1 - v.y0) || 1);
    if (ax.y && ax.y.flip) { sy = -sy; return { sx: sx, tx: p.l - v.x0 * sx, sy: sy, ty: p.t - v.y0 * sy, xlog: xl, ylog: yl }; }
    return { sx: sx, tx: p.l - v.x0 * sx, sy: sy, ty: p.b - v.y0 * sy, xlog: xl, ylog: yl };
  };
  Panel.prototype.toData = function (px, py) {
    var t = this.tf(), x = (px - t.tx) / t.sx, y = (py - t.ty) / t.sy;
    return [t.xlog ? Math.pow(10, x) : x, t.ylog ? Math.pow(10, y) : y];
  };
  Panel.prototype.toDataRaw = function (px, py) { var t = this.tf(); return [(px - t.tx) / t.sx, (py - t.ty) / t.sy]; };

  // ---- drawing
  var FONT = 'Inter, "DejaVu Sans", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
  Panel.prototype.colors = function () {
    var d = this.fig.dark;
    return { bg: d ? "#000000" : "#ffffff", ink: d ? "#f0efec" : "#0b0b0b", ink2: d ? "#c3c2b7" : "#52514e", muted: d ? "#8a8984" : "#6b6a66",
      grid: d ? "#232322" : "#ecebe7", axis: d ? "#4a4946" : "#c3c2b7" };
  };
  Panel.prototype.render = function () {
    if (!this.w || !this.h || this.hidden) return;
    var t0 = global.performance ? global.performance.now() : Date.now();
    try {
      this.draw(this.ctx, this.dpr, false);
      this.cover.style.display = "none";
    } catch (e) {
      this.cover.style.display = "flex"; this.cover.textContent = "This panel could not be drawn: " + (e && e.message ? e.message : e);
      if (global.console) global.console.error("spaclone figure:", e);
    }
    this.lastMs = (global.performance ? global.performance.now() : Date.now()) - t0;
    this.fig.renderMs[this.spec.id] = this.lastMs;
    this.renderedView = JSON.parse(JSON.stringify(this.fig.view(this)));
    this.fresh = true;
    this.renders = (this.renders || 0) + 1;
    this.drawOverlay();
  };
  Panel.prototype.drawOverlay = function () {  // selections, the hover ring and linked highlights, over the data
    var ctx = this.octx, dpr = this.dpr || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.over.width, this.over.height);
    if (this.blitted) return;  // the data picture is moved during a gesture; overlays come back when it settles
    var T = this.tf();
    ctx.save();
    if (!this.isMap) { var p = this.plot; ctx.beginPath(); ctx.rect(p.l * dpr, p.t * dpr, (p.r - p.l) * dpr, (p.b - p.t) * dpr); ctx.clip(); }
    this.drawHighlight(ctx, T, dpr, this.visibleLayers());
    ctx.restore();
    this.drawSelection(ctx, T, dpr);
  };
  Panel.prototype.draw = function (ctx, dpr) {
    var C = this.colors(), W = this.canvas.width, H = this.canvas.height, self = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    var layers = this.visibleLayers(), T = this.tf(), drawn = 0;
    if (this.isMap) layers = drawOrder(layers);
    if (!this.isMap) this.drawGrid(ctx, dpr, T, C);
    var p = this.plot, clip = !this.isMap;
    ctx.save();
    if (clip) { ctx.beginPath(); ctx.rect(p.l * dpr, p.t * dpr, (p.r - p.l) * dpr, (p.b - p.t) * dpr); ctx.clip(); }
    var run = [];
    function flush() {
      if (!run.length) return;
      var img = ctx.getImageData(0, 0, W, H), buf = new Uint32Array(img.data.buffer);
      run.forEach(function (L) { drawn += self.rasterPoints(L, buf, W, H, T, dpr); });
      ctx.putImageData(img, 0, 0);
      run = [];
    }
    for (var i = 0; i < layers.length; i++) {
      var L = layers[i];
      if (L.type === "points" && !this.vectorPoints(L, T)) { run.push(L); continue; }
      flush();
      if (L.type === "points") drawn += this.vectorPointsDraw(ctx, L, T, dpr);
      else if (L.type === "image") this.drawImage(ctx, L, T, dpr);
      else if (L.type === "paths") this.drawPaths(ctx, L, T, dpr);
      else if (L.type === "segs") this.drawSegs(ctx, L, T, dpr);
      else if (L.type === "rects") this.drawRects(ctx, L, T, dpr);
      else if (L.type === "text") this.drawText(ctx, L, T, dpr);
      else if (L.type === "links") this.drawLinks(ctx, L, T, dpr);
    }
    flush();
    for (var hlI = 0; hlI < layers.length; hlI++) {
      var HL = layers[hlI];
      if (HL.type === "points" && HL.label_highlight && HL.color && HL.color.kind === "highlight" && !this.vectorPoints(HL, T)) this.highlightLabels(ctx, HL, T, dpr, this.prepare(HL));
    }
    ctx.restore();
    if (!this.isMap) this.drawAxes(ctx, dpr, T, C);
    else if (this.spec.scalebar !== false) this.drawScalebar(ctx, dpr, C);
    this.blitted = false;
    this.drawnPoints = drawn;
    var empty = this.spec.empty_note;
    if (empty && !layers.some(function (L) { return L.type === "points" && L.n > 0; })) { this.noteEl.textContent = empty; }
    else this.noteEl.textContent = this.noteFor(layers);
  };
  function zOf(L) {  // drawing order on a map: filled shapes (tissue), images, links, points, lines, outlines, labels
    if (L.z !== undefined) return L.z;
    if (L.type === "paths" && L.fill) return 0;
    if (L.type === "image") return 0.5;
    if (L.type === "links") return 1.5;
    if (L.type === "points" || (L.type === "rects" && L.fill)) return 2;
    if (L.type === "segs") return 3;
    if (L.type === "text") return 5;
    return 4;
  }
  function drawOrder(layers) {
    return layers.map(function (L, i) { return [zOf(L), i, L]; }).sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; })
      .map(function (t) { return t[2]; });
  }
  Panel.prototype.noteFor = function (layers) {
    var st = this.fig.state, nw = this.spec.notes_when || [];
    for (var q = 0; q < nw.length; q++) if (matches(nw[q].when, st)) return nw[q].text;
    for (var i = 0; i < layers.length; i++) if (layers[i].note && matches(layers[i].note_when, st)) return layers[i].note;
    return "";
  };
  Panel.prototype.sizePx = function (L, T) {  // marker diameter in CSS px
    var s = L.size || { px: 3 };
    var d = s.px !== undefined ? s.px : (s.um || 10) * Math.abs(T.sx);
    if (s.min !== undefined) d = Math.max(d, s.min);
    if (s.max !== undefined) d = Math.min(d, s.max);
    return Math.max(d, 0.5);
  };
  Panel.prototype.vectorPoints = function (L, T) { return !!L.sizes || (this.sizePx(L, T) * this.dpr > 44 && L.shape !== "square"); };
  // colours per point: Uint32 lut + index array; alpha 0 entries are not drawn
  Panel.prototype.prepare = function (L) {
    var f = this.fig, key = L.id, c = L.color || { kind: "const", color: "#888888" }, st = f.state;
    var dynKey = c.kind === "gene" ? (st[c.gene] || "") + "|" + c.genes + "|" + (st._norm || "")
      : c.kind === "highlight" ? JSON.stringify(st[c.by] || []) : "";
    var pr = this.prepared[key];
    if (pr && pr.dyn === dynKey) return pr;
    var n = L.n, idx, lut, order = null, hid = f.hidden[L.hkey || L.id] || {};
    if (c.kind === "cat") {
      idx = f.blob(c.codes);
      lut = new Uint32Array(Math.max(c.colors.length, 1) + 1);
      for (var i = 0; i < c.colors.length; i++) lut[i] = hid[i] ? 0 : u32(c.colors[i]);
      lut[c.colors.length] = c.nan ? u32(c.nan) : 0;
    } else if (c.kind === "num" || c.kind === "gene") {
      var vals, lo, hi, logs = c.scale === "log";
      if (c.kind === "gene") {
        var gv = st[c.gene] ? f.geneValues(c.genes, st[c.gene]) : null;
        vals = gv ? gv.values : new Float32Array(n); lo = 0; hi = gv ? gv.hi : 1;
      } else { vals = f.blob(c.values); lo = c.domain[0]; hi = c.domain[1]; }
      lut = new Uint32Array(257); var base = rampLut(c.ramp);
      lut.set(base); lut[256] = c.nan ? u32(c.nan) : 0;
      idx = new Uint16Array(n);
      var a = logs ? Math.log10(lo) : lo, b = logs ? Math.log10(hi) : hi, span = (b - a) || 1;
      for (var j = 0; j < n; j++) {
        var v = vals[j];
        if (v !== v || (logs && !(v > 0))) { idx[j] = 256; continue; }
        var tt = ((logs ? Math.log10(v) : v) - a) / span;
        idx[j] = tt <= 0 ? 0 : tt >= 1 ? 255 : (tt * 255 + 0.5) | 0;
      }
      if (c.order !== false) order = binOrder(idx, 257, n);  // high values on top (stable counting sort by colour)
    } else if (c.kind === "highlight") {
      var codes = f.blob(c.codes), names = f.table(c.table), chosen = st[c.by] || [], pos = {};
      chosen.forEach(function (nm, k) { pos[nm] = k; });
      var map = {};  // code -> slot
      for (var h = 0; h < names.length; h++) if (pos[names[h]] !== undefined) map[h] = pos[names[h]];
      lut = new Uint32Array(c.colors.length + 2);
      for (var s2 = 0; s2 < c.colors.length; s2++) lut[s2] = u32(c.colors[s2]);
      lut[c.colors.length] = u32(c.other || "#555555", c.other_alpha === undefined ? 1 : c.other_alpha);
      lut[c.colors.length + 1] = 0;
      idx = new Uint16Array(n); var hl = [], rest = [];
      for (var r = 0; r < n; r++) {
        var m = map[codes[r]];
        if (m !== undefined) { idx[r] = Math.min(m, c.colors.length - 1); hl.push(r); } else { idx[r] = c.other ? c.colors.length : c.colors.length + 1; rest.push(r); }
      }
      order = new Uint32Array(rest.concat(hl));
    } else {
      lut = new Uint32Array([u32(c.color || "#888888")]); idx = null;
    }
    if (!order && L.order) order = f.blob(L.order);
    pr = { lut: lut, idx: idx, order: order, dyn: dynKey };
    this.prepared[key] = pr;
    return pr;
  };
  function binOrder(idx, nb, n) {  // indices sorted by bin, missing values (bin nb - 1) first
    var cnt = new Uint32Array(nb + 1), out = new Uint32Array(n), i;
    for (i = 0; i < n; i++) cnt[(idx[i] + 1) % nb + 1]++;
    for (i = 0; i < nb; i++) cnt[i + 1] += cnt[i];
    for (i = 0; i < n; i++) out[cnt[(idx[i] + 1) % nb]++] = i;
    return out;
  }
  Panel.prototype.rasterPoints = function (L, buf, W, H, T, dpr) {
    var f = this.fig, x = f.blob(L.x), y = f.blob(L.y), n = L.n, pr = this.prepare(L);
    var lut = pr.lut, idx = pr.idx, ord = pr.order, alpha = L.alpha === undefined ? 1 : L.alpha;
    var sx = T.sx * dpr, tx = T.tx * dpr, sy = T.sy * dpr, ty = T.ty * dpr;
    var d = this.sizePx(L, T) * dpr, sq = L.shape === "square", edge = L.edge ? L.edge.px * dpr : 0, ecol = L.edge ? u32(L.edge.color) : 0;
    var p = this.plot, cx0 = this.isMap ? 0 : Math.floor(p.l * dpr), cy0 = this.isMap ? 0 : Math.floor(p.t * dpr);
    var cx1 = this.isMap ? W : Math.ceil(p.r * dpr), cy1 = this.isMap ? H : Math.ceil(p.b * dpr);
    var drawn = 0, k, i, col, px, py, base, j, q;
    if (sq) {
      var hs = d / 2;
      for (k = 0; k < n; k++) {
        i = ord ? ord[k] : k; col = idx ? lut[idx[i]] : lut[0];
        if (!(col >>> 24)) continue;
        var fx = x[i] * sx + tx, fy = y[i] * sy + ty;
        var x0 = Math.floor(fx - hs), x1 = Math.max(Math.floor(fx + hs), x0 + 1), y0 = Math.floor(fy - hs), y1 = Math.max(Math.floor(fy + hs), y0 + 1);
        if (x1 <= cx0 || y1 <= cy0 || x0 >= cx1 || y0 >= cy1) continue;
        if (x0 < cx0) x0 = cx0; if (y0 < cy0) y0 = cy0; if (x1 > cx1) x1 = cx1; if (y1 > cy1) y1 = cy1;
        drawn++;
        for (py = y0; py < y1; py++) { base = py * W; for (px = x0; px < x1; px++) buf[base + px] = alpha < 1 ? blend(buf[base + px], col, alpha) : col; }
      }
      return drawn;
    }
    var r = d / 2;
    if (r < 0.62) {  // sub-pixel points: one pixel weighted by its area
      var a1 = Math.min(1, Math.PI * r * r) * alpha;
      for (k = 0; k < n; k++) {
        i = ord ? ord[k] : k; col = idx ? lut[idx[i]] : lut[0];
        if (!(col >>> 24)) continue;
        px = (x[i] * sx + tx) | 0; py = (y[i] * sy + ty) | 0;
        if (px < cx0 || py < cy0 || px >= cx1 || py >= cy1) continue;
        q = py * W + px; buf[q] = blend(buf[q], col, a1); drawn++;
      }
      return drawn;
    }
    var stc = stencil(r, edge, "c"), R = stc.R;
    var fo = new Int32Array(stc.fx.length), eo = new Int32Array(stc.ex.length), bo = new Int32Array(stc.bx.length);
    for (j = 0; j < fo.length; j++) fo[j] = stc.fy[j] * W + stc.fx[j];
    for (j = 0; j < eo.length; j++) eo[j] = stc.ey[j] * W + stc.ex[j];
    for (j = 0; j < bo.length; j++) bo[j] = stc.by[j] * W + stc.bx[j];
    var nf = fo.length, ne = eo.length, nb = bo.length, bw = stc.bw, bk = stc.bk, op = alpha >= 0.999;
    for (k = 0; k < n; k++) {
      i = ord ? ord[k] : k; col = idx ? lut[idx[i]] : lut[0];
      if (!(col >>> 24)) continue;
      px = Math.round(x[i] * sx + tx); py = Math.round(y[i] * sy + ty);
      if (px < cx0 - R || py < cy0 - R || px >= cx1 + R || py >= cy1 + R || px !== px) continue;
      drawn++;
      var aa = (col >>> 24) / 255 * alpha, full = op && aa > 0.999;
      if (px >= cx0 + R && py >= cy0 + R && px < cx1 - R && py < cy1 - R) {
        base = py * W + px;
        if (full) { for (j = 0; j < nf; j++) buf[base + fo[j]] = col; for (j = 0; j < ne; j++) buf[base + eo[j]] = ecol; }
        else { for (j = 0; j < nf; j++) { q = base + fo[j]; buf[q] = blend(buf[q], col, aa); } for (j = 0; j < ne; j++) { q = base + eo[j]; buf[q] = blend(buf[q], ecol, aa); } }
        for (j = 0; j < nb; j++) {
          q = base + bo[j];
          var c2 = edge ? (bk[j] >= 0.5 ? col : ecol) : col;
          buf[q] = blend(buf[q], c2, bw[j] * aa);
        }
      } else {
        for (j = 0; j < nf + ne + nb; j++) {
          var ox, oy, c3, w3;
          if (j < nf) { ox = stc.fx[j]; oy = stc.fy[j]; c3 = col; w3 = aa; }
          else if (j < nf + ne) { ox = stc.ex[j - nf]; oy = stc.ey[j - nf]; c3 = ecol; w3 = aa; }
          else { var jj = j - nf - ne; ox = stc.bx[jj]; oy = stc.by[jj]; c3 = edge ? (bk[jj] >= 0.5 ? col : ecol) : col; w3 = bw[jj] * aa; }
          var qx = px + ox, qy = py + oy;
          if (qx < cx0 || qy < cy0 || qx >= cx1 || qy >= cy1) continue;
          q = qy * W + qx; buf[q] = w3 > 0.999 ? c3 : blend(buf[q], c3, w3);
        }
      }
    }
    return drawn;
  };
  Panel.prototype.vectorPointsDraw = function (ctx, L, T, dpr) {  // big markers when zoomed far in, or one size per point
    var f = this.fig, x = f.blob(L.x), y = f.blob(L.y), n = L.n, pr = this.prepare(L), r0 = this.sizePx(L, T) / 2 * dpr;
    var sizes = L.sizes ? f.blob(L.sizes) : null, groups = {}, drawn = 0, W = this.canvas.width, H = this.canvas.height;
    for (var k = 0; k < n; k++) {
      var i = pr.order ? pr.order[k] : k, col = pr.idx ? pr.lut[pr.idx[i]] : pr.lut[0];
      if (!(col >>> 24)) continue;
      var r = sizes ? Math.max(sizes[i] / 2 * dpr, 0.6) : r0;
      var px = (x[i] * T.sx + T.tx) * dpr, py = (y[i] * T.sy + T.ty) * dpr;
      if (px < -r || py < -r || px > W + r || py > H + r || px !== px || py !== py) continue;
      (groups[col] = groups[col] || []).push(px, py, r); drawn++;
    }
    ctx.globalAlpha = L.alpha === undefined ? 1 : L.alpha;
    for (var g in groups) {
      var v = +g, pts = groups[g];
      ctx.fillStyle = "rgba(" + (v & 255) + "," + ((v >> 8) & 255) + "," + ((v >> 16) & 255) + "," + (((v >>> 24) & 255) / 255) + ")";
      ctx.beginPath();
      for (var m = 0; m < pts.length; m += 3) { ctx.moveTo(pts[m] + pts[m + 2], pts[m + 1]); ctx.arc(pts[m], pts[m + 1], pts[m + 2], 0, 2 * Math.PI); }
      ctx.fill();
      if (L.edge) { ctx.strokeStyle = L.edge.color; ctx.lineWidth = L.edge.px * dpr; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    if (L.label_highlight && L.color && L.color.kind === "highlight") this.highlightLabels(ctx, L, T, dpr, pr);
    return drawn;
  };
  Panel.prototype.highlightLabels = function (ctx, L, T, dpr, pr) {  // names of the searched items next to them
    var f = this.fig, x = f.blob(L.x), y = f.blob(L.y), codes = f.blob(L.color.codes), names = f.table(L.color.table), C2 = this.colors();
    var nc = (L.color.colors || []).length;
    ctx.save(); ctx.font = "600 " + 11.5 * dpr + "px " + FONT; ctx.textBaseline = "bottom"; ctx.textAlign = "left"; ctx.lineJoin = "round";
    for (var i = 0; i < L.n; i++) {
      if (pr.idx[i] >= nc) continue;
      var px = (x[i] * T.sx + T.tx) * dpr + 6 * dpr, py = (y[i] * T.sy + T.ty) * dpr - 4 * dpr, t = String(names[codes[i]]);
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = C2.bg; ctx.strokeText(t, px, py); ctx.fillStyle = C2.ink; ctx.fillText(t, px, py);
    }
    ctx.restore();
  };
  Panel.prototype.screenRings = function (xy, off, T, dpr) {  // display coords -> device px rings
    var rings = [];
    for (var r = 0; r + 1 < off.length; r++) {
      var a = off[r], b = off[r + 1], pts = new Float64Array((b - a) * 2);
      for (var k = a; k < b; k++) { pts[(k - a) * 2] = (xy[2 * k] * T.sx + T.tx) * dpr; pts[(k - a) * 2 + 1] = (xy[2 * k + 1] * T.sy + T.ty) * dpr; }
      rings.push(pts);
    }
    return rings;
  };
  Panel.prototype.pathOf = function (ctx, rings, closed, which) {
    ctx.beginPath();
    for (var r = 0; r < rings.length; r++) {
      if (which && which[r] === 0) continue;
      var p = rings[r]; if (p.length < 4) continue;
      ctx.moveTo(p[0], p[1]);
      for (var k = 2; k < p.length; k += 2) ctx.lineTo(p[k], p[k + 1]);
      if (closed) ctx.closePath();
    }
  };
  Panel.prototype.drawImage = function (ctx, L, T, dpr) {
    var self = this, im = this.fig.images = this.fig.images || {};
    var img = im[L.id];
    if (!img) {
      img = im[L.id] = new global.Image();
      img.onload = function () { self.fig.renderAll(); };
      img.onerror = function () { self.cover.style.display = "flex"; self.cover.textContent = "an embedded image could not be decoded"; };
      img.src = L.src;
    }
    if (!img.complete || !img.naturalWidth) return;
    ctx.save();
    if (L.clip) {
      var cl = (this.spec.layers || []).filter(function (q) { return q.id === L.clip; })[0];
      if (cl) { var rings = this.screenRings(this.fig.blob(cl.xy), this.fig.blob(cl.off), T, dpr); this.pathOf(ctx, rings, true); ctx.clip("evenodd"); }
    }
    var m = L.m;  // image px -> display: X = m0 u + m2 v + m4, Y = m1 u + m3 v + m5
    ctx.setTransform(dpr * T.sx * m[0], dpr * T.sy * m[1], dpr * T.sx * m[2], dpr * T.sy * m[3], dpr * (T.sx * m[4] + T.tx), dpr * (T.sy * m[5] + T.ty));
    ctx.imageSmoothingEnabled = L.smooth !== false;
    if (ctx.imageSmoothingQuality !== undefined) ctx.imageSmoothingQuality = "high";
    ctx.globalAlpha = L.alpha === undefined ? 1 : L.alpha;
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  };
  Panel.prototype.drawPaths = function (ctx, L, T, dpr) {
    var f = this.fig, rings = this.screenRings(f.blob(L.xy), f.blob(L.off), T, dpr);
    var which = L.colors ? null : null;
    ctx.save();
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (L.fill) { this.pathOf(ctx, rings, true); ctx.fillStyle = L.fill; ctx.globalAlpha = L.fill_alpha === undefined ? 1 : L.fill_alpha; ctx.fill("evenodd"); ctx.globalAlpha = 1; }
    if (L.stroke || L.colors) {
      var width = (L.width || 1) * dpr;
      if (L.halo) { this.pathOf(ctx, rings, !!L.closed, which); ctx.strokeStyle = L.halo; ctx.lineWidth = width + 2.2 * dpr; ctx.setLineDash([]); ctx.stroke(); }
      ctx.setLineDash(L.dash ? L.dash.map(function (v) { return v * dpr; }) : []);
      ctx.lineWidth = width; ctx.globalAlpha = L.alpha === undefined ? 1 : L.alpha;
      if (L.colors) {
        var cols = L.colors;
        for (var r = 0; r < rings.length; r++) { this.pathOf(ctx, [rings[r]], !!L.closed); ctx.strokeStyle = cols[r % cols.length]; ctx.stroke(); }
      } else { this.pathOf(ctx, rings, !!L.closed); ctx.strokeStyle = L.stroke; ctx.stroke(); }
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    if (L.arrows) {  // arrow heads at the given vertices, pointing along the path
      var xy = f.blob(L.xy), at = f.blob(L.arrows.at), sz = (L.arrows.size || 6) * dpr;
      ctx.fillStyle = L.arrows.color || L.stroke;
      ctx.beginPath();
      for (var a = 0; a < at.length; a++) {
        var v = at[a], x0 = (xy[2 * v] * T.sx + T.tx) * dpr, y0 = (xy[2 * v + 1] * T.sy + T.ty) * dpr;
        var x1 = (xy[2 * v + 2] * T.sx + T.tx) * dpr, y1 = (xy[2 * v + 3] * T.sy + T.ty) * dpr;
        var dx = x1 - x0, dy = y1 - y0, len = Math.sqrt(dx * dx + dy * dy); if (!(len > 0)) continue;
        dx /= len; dy /= len;
        var cx = L.arrows.pos === "end" ? x1 - dx * sz * 0.5 : (x0 + x1) / 2, cy = L.arrows.pos === "end" ? y1 - dy * sz * 0.5 : (y0 + y1) / 2;
        ctx.moveTo(cx + dx * sz * 0.6, cy + dy * sz * 0.6);
        ctx.lineTo(cx - dx * sz * 0.5 - dy * sz * 0.45, cy - dy * sz * 0.5 + dx * sz * 0.45);
        ctx.lineTo(cx - dx * sz * 0.5 + dy * sz * 0.45, cy - dy * sz * 0.5 - dx * sz * 0.45);
        ctx.closePath();
      }
      ctx.fill();
    }
    ctx.restore();
  };
  Panel.prototype.drawSegs = function (ctx, L, T, dpr) {
    var f = this.fig, xy = f.blob(L.xy), n = L.n, c = L.color || "#888888";
    ctx.save(); ctx.lineWidth = (L.width || 1) * dpr; ctx.lineCap = L.cap || "butt"; ctx.globalAlpha = L.alpha === undefined ? 1 : L.alpha;
    if (L.dash) ctx.setLineDash(L.dash.map(function (v) { return v * dpr; }));
    var codes = typeof c === "object" ? f.blob(c.codes) : null, groups = {};
    for (var i = 0; i < n; i++) {
      var col = codes ? c.colors[codes[i]] : c; if (!col) continue;
      (groups[col] = groups[col] || []).push(i);
    }
    for (var g in groups) {
      ctx.strokeStyle = g; ctx.beginPath();
      groups[g].forEach(function (i) {
        ctx.moveTo((xy[4 * i] * T.sx + T.tx) * dpr, (xy[4 * i + 1] * T.sy + T.ty) * dpr);
        ctx.lineTo((xy[4 * i + 2] * T.sx + T.tx) * dpr, (xy[4 * i + 3] * T.sy + T.ty) * dpr);
      });
      ctx.stroke();
    }
    ctx.restore();
  };
  Panel.prototype.drawLinks = function (ctx, L, T, dpr) {  // minimum-spanning-tree links of highlighted clones (computed here)
    var f = this.fig, chosen = f.state[L.by] || []; if (!chosen.length) return;
    var x = f.blob(L.x), y = f.blob(L.y), codes = f.blob(L.codes), names = f.table(L.table), pos = {};
    chosen.forEach(function (nm, k) { pos[nm] = k; });
    var members = {};
    for (var i = 0; i < L.n; i++) { var k = pos[names[codes[i]]]; if (k !== undefined) (members[k] = members[k] || []).push(i); }
    ctx.save(); ctx.lineWidth = (L.width || 0.8) * dpr; ctx.globalAlpha = L.alpha === undefined ? 0.8 : L.alpha;
    for (var key in members) {
      var idx = members[key], segs = mst(x, y, idx, L.max_um === undefined ? Infinity : (L.max_um === null ? Infinity : L.max_um));
      ctx.strokeStyle = L.colors[Math.min(+key, L.colors.length - 1)]; ctx.beginPath();
      for (var s = 0; s < segs.length; s += 2) {
        var a = segs[s], b = segs[s + 1];
        ctx.moveTo((x[a] * T.sx + T.tx) * dpr, (y[a] * T.sy + T.ty) * dpr); ctx.lineTo((x[b] * T.sx + T.tx) * dpr, (y[b] * T.sy + T.ty) * dpr);
      }
      ctx.stroke();
    }
    ctx.restore();
  };
  function mst(x, y, idx, maxLen) {  // Prim on distinct positions, O(n^2); links longer than maxLen are left out
    var uniq = [], seen = {};
    idx.forEach(function (i) { var k = x[i] + "," + y[i]; if (!seen[k]) { seen[k] = 1; uniq.push(i); } });
    var n = uniq.length, out = []; if (n < 2 || n > 6000) return out;
    var inT = new Uint8Array(n), best = new Float64Array(n).fill(Infinity), from = new Int32Array(n).fill(-1);
    best[0] = 0;
    for (var it = 0; it < n; it++) {
      var u = -1, bd = Infinity;
      for (var v = 0; v < n; v++) if (!inT[v] && best[v] < bd) { bd = best[v]; u = v; }
      if (u < 0) break;
      inT[u] = 1;
      if (from[u] >= 0 && Math.sqrt(bd) <= maxLen) out.push(uniq[from[u]], uniq[u]);
      var xu = x[uniq[u]], yu = y[uniq[u]];
      for (var w = 0; w < n; w++) if (!inT[w]) { var dx = x[uniq[w]] - xu, dy = y[uniq[w]] - yu, d = dx * dx + dy * dy; if (d < best[w]) { best[w] = d; from[w] = u; } }
    }
    return out;
  }
  Panel.prototype.drawRects = function (ctx, L, T, dpr) {
    var f = this.fig, r = f.blob(L.r), n = L.n, fill = L.fill, st = f.state, vals = null, codes = null, lut = null, lo = 0, hi = 1;
    if (fill && typeof fill === "object") {
      if (fill.values) { vals = f.blob(fill.values); lut = rampLut(fill.ramp); lo = fill.domain[0]; hi = fill.domain[1]; }
      else if (fill.codes) codes = f.blob(fill.codes);
    }
    var hid = f.hidden[L.hkey || L.id] || {};
    ctx.save();
    var labels = L.labels ? f.table(L.labels) : null;
    for (var i = 0; i < n; i++) {
      var x0 = (r[4 * i] * T.sx + T.tx) * dpr, y0 = (r[4 * i + 1] * T.sy + T.ty) * dpr, x1 = (r[4 * i + 2] * T.sx + T.tx) * dpr, y1 = (r[4 * i + 3] * T.sy + T.ty) * dpr;
      var xa = Math.min(x0, x1), ya = Math.min(y0, y1), w = Math.abs(x1 - x0), h = Math.abs(y1 - y0), col = null;
      if (vals) { var v = vals[i]; if (v === v) col = lutHex(lut, clamp(Math.round((v - lo) / ((hi - lo) || 1) * 255), 0, 255)); else col = L.nan || null; }
      else if (codes) { if (hid[codes[i]]) continue; col = fill.colors[codes[i]]; }
      else if (typeof fill === "string") col = fill;
      if (col) { ctx.fillStyle = col; ctx.fillRect(xa, ya, w, h); }
      if (L.stroke) { ctx.strokeStyle = L.stroke; ctx.lineWidth = (L.width || 1) * dpr; ctx.strokeRect(xa, ya, w, h); }
      if (labels && labels[i] && w > 14 * dpr && h > 9 * dpr) {
        ctx.font = (L.label_size || 10.5) * dpr + "px " + FONT; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = L.label_color && L.label_color !== "auto" ? L.label_color : inkOn(col || "#000000");
        if (ctx.measureText(labels[i]).width < w - 4 * dpr) ctx.fillText(labels[i], xa + w / 2, ya + h / 2);
      }
    }
    void st;
    ctx.restore();
  };
  function inkOn(hex) { var c = hexRgb(hex).map(function (v) { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.3 ? "#0b0b0b" : "#ffffff"; }
  Panel.prototype.drawText = function (ctx, L, T, dpr) {
    var f = this.fig, xy = f.blob(L.xy), texts = f.table(L.text), C = this.colors();
    ctx.save();
    ctx.font = (L.weight || "600") + " " + ((L.size || 12) * dpr) + "px " + FONT;
    ctx.textAlign = L.anchor === "start" ? "left" : L.anchor === "end" ? "right" : "center";
    ctx.textBaseline = L.baseline || "middle";
    var boxes = [], fs = (L.size || 12) * dpr;
    for (var i = 0; i < texts.length; i++) {
      var x = (xy[2 * i] * T.sx + T.tx) * dpr + (L.dx || 0) * dpr, y = (xy[2 * i + 1] * T.sy + T.ty) * dpr + (L.dy || 0) * dpr;
      if (L.declutter) {  // a label that would cover one drawn before it waits for a closer zoom
        var w = ctx.measureText(texts[i]).width, x0 = ctx.textAlign === "left" ? x : ctx.textAlign === "right" ? x - w : x - w / 2;
        var y0 = ctx.textBaseline === "bottom" ? y - fs : ctx.textBaseline === "top" ? y : y - fs / 2, hit = false;
        for (var b2 = 0; b2 < boxes.length; b2++) { var q = boxes[b2]; if (x0 < q[2] && x0 + w > q[0] && y0 < q[3] && y0 + fs > q[1]) { hit = true; break; } }
        if (hit) continue;
        boxes.push([x0 - 2 * dpr, y0 - 1 * dpr, x0 + w + 2 * dpr, y0 + fs + 1 * dpr]);
      }
      if (L.halo !== null) { ctx.lineWidth = 3 * dpr; ctx.strokeStyle = L.halo || C.bg; ctx.lineJoin = "round"; ctx.strokeText(texts[i], x, y); }
      ctx.fillStyle = L.color || C.ink; ctx.fillText(texts[i], x, y);
    }
    ctx.restore();
  };
  Panel.prototype.drawHighlight = function (ctx, T, dpr, layers) {
    var hl = this.fig.highlight, f = this.fig, C = this.colors(), self = this; this.hlCount = 0; if (!hl) return;
    layers.forEach(function (L) {
      if (L.type !== "points" || !L.key || L.key_group !== hl.group) return;
      var keys = f.blob(L.key), x = f.blob(L.x), y = f.blob(L.y);
      ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6 * dpr; ctx.beginPath();
      for (var i = 0; i < L.n; i++) if (keys[i] === hl.key) {
        self.hlCount++;
        var px = (x[i] * T.sx + T.tx) * dpr, py = (y[i] * T.sy + T.ty) * dpr;
        ctx.moveTo(px + 6 * dpr, py); ctx.arc(px, py, 6 * dpr, 0, 2 * Math.PI);
      }
      ctx.stroke(); ctx.restore();
    });
    if (this.hoverPt) {
      ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5 * dpr; ctx.beginPath();
      ctx.arc(this.hoverPt[0] * dpr, this.hoverPt[1] * dpr, (this.hoverPt[2] + 3) * dpr, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    }
  };
  Panel.prototype.drawSelection = function (ctx, T, dpr) {
    var sel = this.fig.selection, C = this.colors(), drag = this.drag;
    ctx.save();
    if (sel && sel.panel === this.spec.id) {
      ctx.beginPath();
      for (var i = 0; i < sel.xs.length; i++) { var px = (sel.xs[i] * T.sx + T.tx) * dpr, py = (sel.ys[i] * T.sy + T.ty) * dpr; if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.closePath(); ctx.fillStyle = rgba(C.ink, 0.06); ctx.fill();
      ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3 * dpr; ctx.stroke();
    }
    if (drag && drag.shape && drag.shape.length > 1) {
      ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2 * dpr; ctx.beginPath();
      drag.shape.forEach(function (p, k) { if (k) ctx.lineTo(p[0] * dpr, p[1] * dpr); else ctx.moveTo(p[0] * dpr, p[1] * dpr); });
      if (drag.kind !== "lasso") ctx.closePath();
      ctx.fillStyle = rgba(C.ink, drag.kind === "zoom" ? 0.08 : 0.05); ctx.fill(); ctx.stroke();
    }
    ctx.restore();
  };
  Panel.prototype.drawScalebar = function (ctx, dpr, C) {
    var v = this.fig.view(this), um = niceLength(this.w / v.k / 5); if (!um) return;
    var len = um * v.k, x0 = 14, y0 = this.h - 14, label = fmtLen(um);
    ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = "11.5px " + FONT;
    var tw = ctx.measureText(label).width, pw = Math.max(len, tw) + 12, px0 = x0 + len / 2 - pw / 2;
    ctx.fillStyle = rgba(C.bg, 0.78); ctx.fillRect(Math.min(px0, x0 - 6), y0 - 22, Math.max(pw, len + 12), 30);  // the plate
    ctx.lineWidth = 2; ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + len, y0); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillStyle = C.ink; ctx.fillText(label, x0 + len / 2, y0 - 4);
    ctx.restore();
    this.scalebarUm = um; this.scalebarPx = len;
  };
  Panel.prototype.ticks = function (a, lo, hi, n) {
    if (a.scale === "cat") {
      var out = [], cats = a.cats || [], span = Math.abs(hi - lo) || 1;
      var room = (a === (this.spec.axes || {}).y ? (this.plot.b - this.plot.t) : (this.plot.r - this.plot.l)) / span;
      var step = Math.max(1, Math.ceil(13 / Math.max(room, 1e-9)));
      for (var i = 0; i < cats.length; i += step) if (i >= lo - 0.5 && i <= hi + 0.5) out.push(i);
      return out;
    }
    if (a.ticks) return a.ticks.map(function (t) { return a.scale === "log" ? Math.log10(t) : t; }).filter(function (t) { return t >= lo - 1e-9 && t <= hi + 1e-9; });
    return a.scale === "log" ? logTicks(lo, hi) : linTicks(lo, hi, n);
  };
  Panel.prototype.tickLabel = function (a, t) {
    if (a.scale === "cat") return String((a.cats || [])[Math.round(t)]);
    if (a.ticks && a.ticklabels) { var i = a.ticks.indexOf(a.scale === "log" ? +Math.pow(10, t).toPrecision(12) : t); if (i >= 0) return a.ticklabels[i]; }
    var v = a.scale === "log" ? Math.pow(10, t) : t;
    if (a.scale === "log") v = +v.toPrecision(6);
    return fmtNum(v, a.fmt);
  };
  Panel.prototype.drawGrid = function (ctx, dpr, T, C) {
    var ax = this.spec.axes || {}, v = this.fig.view(this), p = this.plot;
    ctx.save(); ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    var self = this;
    if (ax.x && ax.x.grid) this.ticks(ax.x, v.x0, v.x1, Math.max(3, (p.r - p.l) / 90)).forEach(function (t) {
      var x = Math.round((t * T.sx + T.tx) * dpr) + 0.5; ctx.beginPath(); ctx.moveTo(x, p.t * dpr); ctx.lineTo(x, p.b * dpr); ctx.stroke();
    });
    if (ax.y && ax.y.grid) this.ticks(ax.y, v.y0, v.y1, Math.max(3, (p.b - p.t) / 60)).forEach(function (t) {
      var y = Math.round((t * T.sy + T.ty) * dpr) + 0.5; ctx.beginPath(); ctx.moveTo(p.l * dpr, y); ctx.lineTo(p.r * dpr, y); ctx.stroke();
    });
    (this.spec.refs || []).forEach(function (r) {  // reference lines (chance level, thresholds)
      if (!matches(r.when, self.fig.state)) return;
      ctx.strokeStyle = r.color || C.ink2; ctx.lineWidth = (r.width || 1) * dpr; ctx.setLineDash(r.dash ? r.dash.map(function (d) { return d * dpr; }) : []);
      ctx.beginPath();
      if (r.x !== undefined) { var xv = ax.x && ax.x.scale === "log" ? Math.log10(r.x) : r.x, x = (xv * T.sx + T.tx) * dpr; ctx.moveTo(x, p.t * dpr); ctx.lineTo(x, p.b * dpr); }
      else { var yv = ax.y && ax.y.scale === "log" ? Math.log10(r.y) : r.y, y = (yv * T.sy + T.ty) * dpr; ctx.moveTo(p.l * dpr, y); ctx.lineTo(p.r * dpr, y); }
      ctx.stroke(); ctx.setLineDash([]);
    });
    ctx.restore();
  };
  Panel.prototype.drawAxes = function (ctx, dpr, T, C) {
    var ax = this.spec.axes || {}, v = this.fig.view(this), p = this.plot, self = this;
    ctx.save(); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1; ctx.fillStyle = C.ink2; ctx.font = "11.5px " + FONT;
    if (ax.x && !ax.x.hide) {
      ctx.beginPath(); ctx.moveTo(p.l, p.b + 0.5); ctx.lineTo(p.r, p.b + 0.5); ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      var lastR = -Infinity;
      this.ticks(ax.x, v.x0, v.x1, Math.max(3, (p.r - p.l) / 90)).forEach(function (t) {
        var x = t * T.sx + T.tx; if (x < p.l - 1 || x > p.r + 1) return;
        ctx.beginPath(); ctx.moveTo(x, p.b); ctx.lineTo(x, p.b + 4); ctx.stroke();
        var lab = self.tickLabel(ax.x, t);
        if (ax.x.rotate) {
          ctx.save(); ctx.translate(x, p.b + 7); ctx.rotate(-Math.PI / 4); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(lab, 0, 0); ctx.restore();
        } else {
          var w = ctx.measureText(lab).width; if (x - w / 2 < lastR + 6) return;
          ctx.fillText(lab, x, p.b + 6); lastR = x + w / 2;
        }
      });
      if (ax.x.label) { ctx.fillStyle = C.ink; ctx.font = "12px " + FONT; ctx.textBaseline = "bottom"; ctx.fillText(ax.x.label, (p.l + p.r) / 2, Math.min(this.h, p.b + (p.bm || 0)) - 4); ctx.fillStyle = C.ink2; ctx.font = "11.5px " + FONT; }
    }
    if (ax.y && !ax.y.hide) {
      ctx.beginPath(); ctx.moveTo(p.l - 0.5, p.t); ctx.lineTo(p.l - 0.5, p.b); ctx.stroke();
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      var lastY = Infinity;
      this.ticks(ax.y, v.y0, v.y1, Math.max(3, (p.b - p.t) / 60)).forEach(function (t) {
        var y = t * T.sy + T.ty; if (y < p.t - 1 || y > p.b + 1) return;
        if (Math.abs(y - lastY) < 13) return; lastY = y;
        ctx.beginPath(); ctx.moveTo(p.l - 4, y); ctx.lineTo(p.l, y); ctx.stroke();
        ctx.fillText(self.tickLabel(ax.y, t), p.l - 7, y);
      });
      if (ax.y.label) {
        ctx.save(); ctx.fillStyle = C.ink; ctx.font = "12px " + FONT; ctx.translate(Math.max(13, p.l - (p.lm || 0) + 13), (p.t + p.b) / 2); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(ax.y.label, 0, 0); ctx.restore();
      }
    }
    ctx.restore();
  };

  // ---- interaction
  Panel.prototype.bind = function () {
    var self = this, cv = this.canvas, f = this.fig;
    cv.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      cv.setPointerCapture && cv.setPointerCapture(e.pointerId);
      var pt = self.local(e), kind = f.tool;
      if (kind === "pan" && e.shiftKey) kind = "zoom";
      self.pointers = self.pointers || {};
      self.pointers[e.pointerId] = pt;
      if (Object.keys(self.pointers).length === 2) { self.startPinch(); return; }
      var v = f.view(self);
      self.drag = { kind: kind, x: pt[0], y: pt[1], moved: false, v: JSON.parse(JSON.stringify(v)), shape: kind === "pan" ? null : [pt] };
      self.hideTip();
      if (kind === "pan") self.el.classList.add("spc-dragging");
      f.panels.forEach(function (p) { if (p === self || (self.spec.link && p.spec.link === self.spec.link)) p.ensureSnapshot(); });
    });
    cv.addEventListener("pointermove", function (e) {
      var pt = self.local(e);
      if (self.pointers && self.pointers[e.pointerId]) self.pointers[e.pointerId] = pt;
      if (self.pinch) { self.movePinch(); return; }
      var d = self.drag;
      if (!d) { self.queueHover(pt); return; }
      if (Math.abs(pt[0] - d.x) + Math.abs(pt[1] - d.y) > 3) d.moved = true;
      if (d.kind === "pan") { self.panTo(d, pt); return; }
      if (d.kind === "lasso") d.shape.push(pt);
      else d.shape = [[d.x, d.y], [pt[0], d.y], [pt[0], pt[1]], [d.x, pt[1]]];
      self.overlayFrame();
    });
    function end(e) {
      if (self.pointers) delete self.pointers[e.pointerId];
      if (self.pinch) { if (!self.pointers || Object.keys(self.pointers).length < 2) { self.pinch = null; self.settle(); } return; }
      var d = self.drag; self.drag = null; self.el.classList.remove("spc-dragging");
      if (!d) return;
      if (d.kind === "pan") { if (d.moved) self.settle(); else self.click(self.local(e)); return; }
      if (!d.moved || !d.shape || d.shape.length < 3) { self.drawOverlay(); return; }
      if (d.kind === "zoom") self.zoomToScreenBox(d.shape);
      else self.selectShape(d.shape, d.kind !== "lasso");
    }
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.addEventListener("pointerleave", function () { if (!self.drag) self.hideTip(); });
    cv.addEventListener("dblclick", function (e) { e.preventDefault(); self.resetView(); });
    cv.addEventListener("wheel", function (e) {
      if (!f.active && !e.ctrlKey && !e.metaKey) { self.showHint("click the figure first, or hold Ctrl / \u2318, to zoom with the wheel"); return; }
      e.preventDefault();
      var pt = self.local(e), dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaMode === 2 ? e.deltaY * 400 : e.deltaY;
      self.zoomAt(pt, Math.exp(-clamp(dy, -250, 250) * 0.0022));
    }, { passive: false });
    this.el.addEventListener("keydown", function (e) {
      var k = e.key, v = f.view(self), st = 60;
      if (k === "+" || k === "=") self.zoomAt([self.w / 2, self.h / 2], 1.4);
      else if (k === "-" || k === "_") self.zoomAt([self.w / 2, self.h / 2], 1 / 1.4);
      else if (k === "0") self.resetView();
      else if (k === "Escape") { f.selection = null; f.updateStats(); f.overlays(); }
      else if (k.indexOf("Arrow") === 0) {
        var dx = k === "ArrowLeft" ? -st : k === "ArrowRight" ? st : 0, dy = k === "ArrowUp" ? -st : k === "ArrowDown" ? st : 0;
        var d0 = { v: JSON.parse(JSON.stringify(v)), x: 0, y: 0 }; self.panTo(d0, [-dx, -dy]); self.settle();
      } else return;
      e.preventDefault();
    });
  };
  Panel.prototype.local = function (e) { var r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  Panel.prototype.showHint = function (text) {
    var h = this.hint; h.textContent = text; h.classList.add("spc-on");
    global.clearTimeout(this._hintT); this._hintT = global.setTimeout(function () { h.classList.remove("spc-on"); }, 1400);
  };
  Panel.prototype.ensureSnapshot = function () {  // a copy of the last full drawing and the view it was drawn at
    if (this.snapView && this.snapRenders === this.renders) return true;
    if (!this.fresh) return !!this.snapView;  // the canvas holds a moved picture: keep the snapshot it came from
    if (!this.snap) this.snap = DOC.createElement("canvas");
    if (this.snap.width !== this.canvas.width || this.snap.height !== this.canvas.height) { this.snap.width = this.canvas.width; this.snap.height = this.canvas.height; }
    var c = this.snap.getContext("2d"); c.setTransform(1, 0, 0, 1, 0, 0); c.drawImage(this.canvas, 0, 0);
    this.snapView = this.renderedView; this.snapRenders = this.renders;
    return true;
  };
  Panel.prototype.panTo = function (d, pt) {
    var v = this.fig.view(this), dx = pt[0] - d.x, dy = pt[1] - d.y;
    if ((this.spec.axes || {}).zoom === "none") return;
    v.atHome = false;
    if (this.isMap) {
      var inv = this.spec.invert_y !== false;
      v.cx = d.v.cx - dx / v.k; v.cy = inv ? d.v.cy - dy / v.k : d.v.cy + dy / v.k;
    } else {
      var T0 = this.tf(), zx = (this.spec.axes || {}).zoom || "xy";
      var ddx = dx / T0.sx, ddy = dy / T0.sy;
      if (zx.indexOf("x") >= 0) { v.x0 = d.v.x0 - ddx; v.x1 = d.v.x1 - ddx; }
      if (zx.indexOf("y") >= 0) { v.y0 = d.v.y0 - ddy; v.y1 = d.v.y1 - ddy; }
    }
    this.frame();
  };
  Panel.prototype.zoomAt = function (pt, factor) {
    var v = this.fig.view(this), ax = this.spec.axes || {}, me = this;
    if (ax.zoom === "none") return;
    this.fig.panels.forEach(function (p) { if (p === me || (me.spec.link && p.spec.link === me.spec.link)) p.ensureSnapshot(); });
    v.atHome = false;
    if (this.isMap) {
      var d = this.toDataRaw(pt[0], pt[1]), k2 = clamp(v.k * factor, (v.k0 || v.k) * 0.25, (v.k0 || v.k) * 4000);
      var inv = this.spec.invert_y !== false;
      v.cx = d[0] - (pt[0] - this.w / 2) / k2;
      v.cy = inv ? d[1] - (pt[1] - this.h / 2) / k2 : d[1] + (pt[1] - this.h / 2) / k2;
      v.k = k2;
    } else {
      var zx = ax.zoom || "xy", dd = this.toDataRaw(pt[0], pt[1]);
      if (zx.indexOf("x") >= 0) { v.x0 = dd[0] + (v.x0 - dd[0]) / factor; v.x1 = dd[0] + (v.x1 - dd[0]) / factor; }
      if (zx.indexOf("y") >= 0) { v.y0 = dd[1] + (v.y0 - dd[1]) / factor; v.y1 = dd[1] + (v.y1 - dd[1]) / factor; }
    }
    this.frame();
    var self = this; global.clearTimeout(this._settleT); this._settleT = global.setTimeout(function () { self.settle(); }, 140);
  };
  Panel.prototype.zoomToScreenBox = function (shape) {
    var xs = shape.map(function (p) { return p[0]; }), ys = shape.map(function (p) { return p[1]; });
    var a = this.toDataRaw(Math.min.apply(null, xs), Math.min.apply(null, ys)), b = this.toDataRaw(Math.max.apply(null, xs), Math.max.apply(null, ys));
    this.zoomToBox([Math.min(a[0], b[0]), Math.max(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[1], b[1])]);
  };
  Panel.prototype.zoomToBox = function (box) {  // box in display (raw axis) units: x0, x1, y0, y1
    var v = this.fig.view(this);
    v.atHome = false;
    if (this.isMap) {
      var bw = Math.max(box[1] - box[0], 1e-6), bh = Math.max(box[3] - box[2], 1e-6);
      v.k = clamp(Math.min(this.w / bw, this.h / bh), (v.k0 || v.k) * 0.25, (v.k0 || v.k) * 4000);
      v.cx = (box[0] + box[1]) / 2; v.cy = (box[2] + box[3]) / 2;
    } else {
      var zx = (this.spec.axes || {}).zoom || "xy";
      if (zx.indexOf("x") >= 0) { v.x0 = box[0]; v.x1 = box[1]; }
      if (zx.indexOf("y") >= 0) { v.y0 = box[2]; v.y1 = box[3]; }
    }
    this.fig.renderLinked(this);
  };
  Panel.prototype.frame = function () {  // fluid update: full redraw when fast, otherwise the last picture moved
    var self = this, f = this.fig;
    if (this._raf) return;
    this._raf = global.requestAnimationFrame(function () {
      self._raf = null;
      var slow = 0;
      f.panels.forEach(function (p) { if (p === self || (self.spec.link && p.spec.link === self.spec.link)) slow = Math.max(slow, p.lastMs); });
      if (slow < 28) { f.renderLinked(self); return; }
      f.panels.forEach(function (p) { if (p === self || (self.spec.link && p.spec.link === self.spec.link)) p.blitSnapshot(); });
      global.clearTimeout(self._settleT); self._settleT = global.setTimeout(function () { self.settle(); }, 160);
    });
  };
  Panel.prototype.blitSnapshot = function () {
    var v = this.fig.view(this), ctx = this.ctx, dpr = this.dpr, C = this.colors();
    if (!this.isMap || !this.ensureSnapshot()) return this.render();
    var s = this.snapView;
    if (!s || !s.k) return this.render();
    var scale = v.k / s.k, inv = this.spec.invert_y !== false;
    var ox = (this.w / 2 - s.cx * s.k), oy = inv ? (this.h / 2 - s.cy * s.k) : (this.h / 2 + s.cy * s.k);
    var nx = (this.w / 2 - v.cx * v.k), ny = inv ? (this.h / 2 - v.cy * v.k) : (this.h / 2 + v.cy * v.k);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(scale, 0, 0, scale, (nx - ox * scale) * dpr, (ny - oy * scale) * dpr);
    ctx.drawImage(this.snap, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.fresh = false; this.blitted = true; this.drawOverlay();
  };
  Panel.prototype.resetView = function () {
    var f = this.fig, self = this;
    f.panels.forEach(function (p) { if (p === self || (self.spec.link && p.spec.link === self.spec.link)) { var v = f.view(p); for (var k in v) delete v[k]; } });
    this.home(); f.renderLinked(this);
  };
  Panel.prototype.settle = function () { global.clearTimeout(this._settleT); this.fig.renderLinked(this); };
  Panel.prototype.overlayFrame = function () {
    var self = this;
    if (this._oraf) return;
    this._oraf = global.requestAnimationFrame(function () { self._oraf = null; self.drawOverlay(); });
  };
  Panel.prototype.startPinch = function () {
    var ids = Object.keys(this.pointers), a = this.pointers[ids[0]], b = this.pointers[ids[1]];
    this.drag = null;
    this.pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], v: JSON.parse(JSON.stringify(this.fig.view(this))) };
    this.ensureSnapshot();
  };
  Panel.prototype.movePinch = function () {
    var ids = Object.keys(this.pointers); if (ids.length < 2) return;
    var a = this.pointers[ids[0]], b = this.pointers[ids[1]], p = this.pinch, v = this.fig.view(this);
    var d = Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    for (var key in p.v) v[key] = p.v[key];
    this.panTo({ v: p.v, x: p.c[0], y: p.c[1] }, c);
    this.zoomAt(c, d / p.d);
  };
  Panel.prototype.click = function (pt) {
    var hit = this.hitTest(pt), f = this.fig;
    if (hit && hit.action && hit.action.zoom) { this.zoomToBox(hit.action.zoom); return; }
    if (hit && hit.action && hit.action.set) { f.setState(hit.action.set[0], hit.action.set[1]); var ctl = f["ctl_" + hit.action.set[0]]; if (ctl) ctl.set(hit.action.set[1]); return; }
    if (f.selection && f.selection.panel === this.spec.id && !hit) { /* a plain click keeps the selection */ }
  };
  Panel.prototype.selectShape = function (shape, isBox) {
    var self = this, xs = [], ys = [];
    shape.forEach(function (p) { var d = self.toDataRaw(p[0], p[1]); xs.push(d[0]); ys.push(d[1]); });
    if (isBox) {
      var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
      xs = [x0, x1, x1, x0]; ys = [y0, y0, y1, y1];
    }
    this.fig.selection = { panel: this.spec.id, xs: xs, ys: ys, box: !!isBox };
    this.fig.updateStats(); this.fig.overlays();
  };
  // ---- hover
  Panel.prototype.queueHover = function (pt) {
    this._hpt = pt; var self = this;
    if (this._hraf) return;
    this._hraf = global.requestAnimationFrame(function () { self._hraf = null; self.hover(self._hpt); });
  };
  Panel.prototype.hitTest = function (pt) {
    var layers = this.visibleLayers(), T = this.tf(), f = this.fig, d = this.toDataRaw(pt[0], pt[1]);
    if (!this.isMap) { var p = this.plot; if (pt[0] < p.l || pt[0] > p.r || pt[1] < p.t || pt[1] > p.b) return null; }
    for (var li = layers.length - 1; li >= 0; li--) {
      var L = layers[li];
      if (L.hover === false) continue;
      if (L.type === "points" && L.n) {
        var x = f.blob(L.x), y = f.blob(L.y), pr = this.prepare(L);
        var idx = this.indexes[L.id] || (this.indexes[L.id] = new GridIndex(x, y, L.n));
        var rpx = Math.max((L.sizes ? 16 : this.sizePx(L, T)) / 2 + 3, 6), rr = rpx / Math.abs(T.sx), self = this;
        var i = idx.nearest(x, y, d[0], d[1], Math.max(rr, rpx / Math.abs(T.sy)), function (j) { var c = pr.idx ? pr.lut[pr.idx[j]] : pr.lut[0]; return (c >>> 24) > 0 && (!L.hover_only_highlight || pr.idx[j] < (L.color.colors || []).length); });
        void self;
        if (i >= 0) return { layer: L, index: i, px: [x[i] * T.sx + T.tx, y[i] * T.sy + T.ty, this.sizePx(L, T) / 2] };
      } else if (L.type === "rects" && L.hover) {
        var r = f.blob(L.r);
        for (var k = L.n - 1; k >= 0; k--) {
          if (d[0] >= Math.min(r[4 * k], r[4 * k + 2]) && d[0] <= Math.max(r[4 * k], r[4 * k + 2]) && d[1] >= Math.min(r[4 * k + 1], r[4 * k + 3]) && d[1] <= Math.max(r[4 * k + 1], r[4 * k + 3]))
            return { layer: L, index: k, action: L.actions ? L.actions[k] : null };
        }
      } else if (L.type === "paths" && L.hover && L.closed) {
        var xy = f.blob(L.xy), off = f.blob(L.off);
        for (var q = 0; q + 1 < off.length; q++) {
          var xs = [], ys = [];
          for (var m = off[q]; m < off[q + 1]; m++) { xs.push(xy[2 * m]); ys.push(xy[2 * m + 1]); }
          if (xs.length > 2 && pip(d[0], d[1], xs, ys)) return { layer: L, index: q, action: L.actions ? L.actions[q] : null };
        }
      } else if (L.type === "image" && L.values) {
        var iv = L.inv, u = iv[0] * d[0] + iv[2] * d[1] + iv[4], w = iv[1] * d[0] + iv[3] * d[1] + iv[5];
        var ui = Math.floor(u), vi = Math.floor(w);
        if (ui >= 0 && vi >= 0 && ui < L.values.w && vi < L.values.h) {
          var vals = f.blob(L.values.blob), val = vals[vi * L.values.w + ui];
          if (val === val) return { layer: L, value: val };
        }
      }
    }
    return null;
  };
  Panel.prototype.hover = function (pt) {
    var hit = this.hitTest(pt), f = this.fig;
    if (!hit) { this.hideTip(); return; }
    var L = hit.layer, H = L.hover || {}, rows = [], title = "";
    if (hit.index !== undefined) {
      var i = hit.index;
      if (H.title) title = this.fieldText(H.title, i);
      (H.rows || []).forEach(function (r) { var t = this.fieldText(r, i); if (t !== null && t !== "") rows.push([r.label, t]); }, this);
      if (L.type === "points" && L.color && (L.color.kind === "gene") && f.state[L.color.gene]) {
        var gv = f.geneValues(L.color.genes, f.state[L.color.gene]);
        if (gv) rows.push([gv.name + " (" + gv.unit + ")", fmtNum(gv.values[i])]);
      }
      if (L.key && L.key_group) {
        var kv = f.blob(L.key)[i];
        if (!f.highlight || f.highlight.key !== kv || f.highlight.group !== L.key_group) {
          f.highlight = { group: L.key_group, key: kv };
          f.panels.forEach(function (p) { p.drawOverlay(); });
        }
      }
    } else if (hit.value !== undefined) {
      title = H.title || L.name || "";
      rows.push([L.values.label || "value", fmtNum(hit.value, L.values.fmt)]);
    }
    this.hoverPt = hit.px || null;
    this.drawOverlay();
    this.showTip(pt, title, rows);
  };
  Panel.prototype.fieldText = function (r, i) {
    var f = this.fig;
    if (typeof r === "string") return r;
    if (r.text) return r.text;
    if (r.table !== undefined) { var codes = f.blob(r.codes), tb = f.table(r.table), c = codes[i]; return c === r.nan ? null : (tb[c] === undefined ? null : String(tb[c])); }
    if (r.values) { var v = f.blob(r.values)[i]; return v === v ? fmtNum(v, r.fmt) : (r.keep_nan ? "\u2013" : null); }
    if (r.list) return r.list[i] === undefined ? null : String(r.list[i]);
    return null;
  };
  Panel.prototype.showTip = function (pt, title, rows) {
    var tip = this.fig.tip, html = "";
    if (title) html += '<div class="spc-tip-h">' + esc(title) + "</div>";
    if (rows.length) { html += "<table>"; rows.forEach(function (r) { html += "<tr><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td></tr>"; }); html += "</table>"; }
    if (!html) { this.hideTip(); return; }
    tip.innerHTML = html; tip.style.display = "block";
    var rr = this.fig.root.getBoundingClientRect(), pr = this.el.getBoundingClientRect();
    var x = pr.left - rr.left + pt[0] + 14, y = pr.top - rr.top + pt[1] + 14, tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > rr.width - 4) x = pr.left - rr.left + pt[0] - tw - 12;
    if (y + th > rr.height - 4) y = pr.top - rr.top + pt[1] - th - 12;
    tip.style.left = Math.max(2, x) + "px"; tip.style.top = Math.max(2, y) + "px";
  };
  Panel.prototype.hideTip = function () {
    this.fig.tip.style.display = "none";
    var f = this.fig, had = this.hoverPt;
    this.hoverPt = null;
    if (f.highlight) { f.highlight = null; f.panels.forEach(function (p) { p.drawOverlay(); }); } else if (had) this.drawOverlay();
  };

  // ------------------------------------------------------------------------------------------------ mounting
  function mount(root) {
    if (root.__spc) return root.__spc;
    var specEl = root.querySelector("script.spc-spec") || (root.id && DOC.getElementById(root.id + "-spec"));
    if (!specEl) return null;
    try {
      if (!global.HTMLCanvasElement || !DOC.createElement("canvas").getContext) throw new Error("this browser has no canvas");
      var spec = JSON.parse(specEl.textContent);
      var fig = new Figure(root, spec);
      root.__spc = fig;
      root.setAttribute("data-spc-ready", "1");
      engines[VERSION].figures.push(fig);
      try { if (global.parent && global.parent !== global) global.parent.postMessage({ spaclone: "ready", id: root.id, height: root.offsetHeight }, "*"); } catch (e) { /* cross-origin parent */ }
      return fig;
    } catch (e) {
      var reason = e && e.message ? e.message : String(e);
      root.setAttribute("data-spc-ready", "error"); root.setAttribute("data-spc-error", reason);
      Array.prototype.slice.call(root.children).forEach(function (c) {  // drop whatever was half built
        if (c.tagName === "SCRIPT" || c.tagName === "NOSCRIPT" || /(^| )spc-fallback( |$)/.test(c.className || "")) return;
        root.removeChild(c);
      });
      var fb = root.querySelector(".spc-fallback"); if (fb) fb.style.display = "";
      el("div", "spc-error", root, "The interactive figure could not start (" + reason + ")" + (fb ? "; the picture above is its static view." : "."));
      try { if (global.parent && global.parent !== global) global.parent.postMessage({ spaclone: "error", id: root.id, message: reason }, "*"); } catch (e2) { /* cross-origin parent */ }
      if (global.console) global.console.error("spaclone figure:", e);
      return null;
    }
  }
  function mountAll(script) {
    var sel = '[data-spc-engine="' + VERSION + '"]:not([data-spc-ready])', i;
    var roots = Array.prototype.slice.call(DOC.querySelectorAll(sel));
    // a frontend may run an output's scripts before attaching it to the page: look next to this script as well
    var host = script && script.parentNode;
    if (host && host.querySelectorAll) roots = roots.concat(Array.prototype.slice.call(host.querySelectorAll(sel)));
    for (i = 0; i < roots.length; i++) mount(roots[i]);
  }
  engines[VERSION] = { version: VERSION, mount: mount, mountAll: mountAll, figures: [], inflate: inflateZlib, varints: varints,
    b64: b64bytes, pip: pip, niceLength: niceLength, linTicks: linTicks, decodeBlob: decodeBlob };
  global.SpacloneFigure = engines[VERSION];
  if (DOC.readyState === "loading") DOC.addEventListener("DOMContentLoaded", function () { mountAll(SCRIPT); }); else mountAll(SCRIPT);
})(typeof window !== "undefined" ? window : this);
