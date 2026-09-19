from pathlib import Path
import sys
p=Path(sys.argv[1])/'b12x/moe/_shared/kernels/w4a8_v41_slice.py'
s=p.read_text().replace('grouped=False, atomic_tokens=False, intermediate=576):', 'grouped=False, atomic_tokens=False, intermediate=576, adaptive_sms=0):',1).replace('        self.atomic_tokens = atomic_tokens', '        self.adaptive_sms = adaptive_sms\n        self.atomic_tokens = atomic_tokens',1)
s=s.replace('        groups: Int32 = 1,\n', '        groups: Int32 = 1,\n        active_groups: cute.Tensor | None = None,\n',1)
s=s.replace('self.kernel(x, xs, w13, s13, w2, s2, routing, out, rows, metadata)', 'self.kernel(x, xs, w13, s13, w2, s2, routing, out, rows, metadata, active_groups)')
s=s.replace('grid=(self.slices, groups if self.grouped else 1, 1)', 'grid=(self.slices, groups if self.grouped else 1, 8 if self.adaptive_sms else 1)')
s=s.replace('        metadata: cute.Tensor | None,\n', '        metadata: cute.Tensor | None,\n        active_groups: cute.Tensor | None,\n',1)
s=s.replace('        if active > 0:\n', '''        shards = Int32(1)
        output_shard = cute.arch.block_idx()[2]
        if cutlass.const_expr(self.adaptive_sms > 0):
            tasks = active_groups[0] * Int32(self.slices)
            for factor in cutlass.range_constexpr(2, 9):
                if cutlass.const_expr(40 % factor == 0):
                    if tasks > 0 and tasks * factor <= self.adaptive_sms:
                        shards = Int32(factor)
            active = active * Int32(output_shard < shards)
        if active > 0:
''',1)
s=s.replace('            for ot in range(40):', '            for ot in range(output_shard * (40 // shards), (output_shard + 1) * (40 // shards)):')
p.write_text(s)
