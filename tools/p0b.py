import hashlib
P = "code/xpolicylab/policy/OpenVLA_R42/r42_adapter.py"
src = open(P).read()
b = hashlib.sha256(src.encode()).hexdigest()
assert src.count("arr.shape[-1] not in (3, 4)") == 1
src = src.replace("arr.shape[-1] not in (3, 4)", "arr.shape[-1] != 3")
assert src.count("float(timestamp) < last") == 1
src = src.replace("float(timestamp) < last", "float(timestamp) <= last")
old = "    return cmd.astype(np.float32), r_hat.astype(np.float32), info"
new = chr(10).join([
    "    if limits is not None:",
    "        lo2 = np.asarray(limits[chr(97)+chr(109)+chr(105)+chr(110)], np.float32)",
    "        hi2 = np.asarray(limits[chr(97)+chr(109)+chr(97)+chr(120)], np.float32)",
    "        cmd = np.clip(cmd, lo2, hi2)",
    "        if not np.isfinite(cmd).all():",
    "            raise R42Error(decoded_command_not_finite)",
    "    return cmd.astype(np.float32), r_hat.astype(np.float32), info",
])
assert src.count(old) == 1
src = src.replace(old, new)
open(P, "w").write(src)
print("sha_before", b)
print("sha_after", hashlib.sha256(src.encode()).hexdigest())
