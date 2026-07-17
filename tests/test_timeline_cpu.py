"""CPU tests for the save-DAG timeline registry."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from savepoint.timeline import Timeline  # noqa: E402


def test_fork_tree_and_persistence():
    with tempfile.TemporaryDirectory() as d:
        tl = Timeline(d)
        root = tl.add(path="a.wsave", latent_frame=0, mode="state", label="start")
        mid = tl.add(path="b.wsave", latent_frame=12, mode="state", parent_id=root.save_id)
        # fork: two children of the same save
        left = tl.add(path="c.wsave", latent_frame=24, mode="reprime",
                      parent_id=mid.save_id, label="took the shortcut")
        right = tl.add(path="d.wsave", latent_frame=24, mode="reprime",
                       parent_id=mid.save_id, label="stayed on the road")
        assert [n.save_id for n in tl.children(mid.save_id)] == [left.save_id, right.save_id]
        assert [n.save_id for n in tl.lineage(left.save_id)] == \
               [root.save_id, mid.save_id, left.save_id]

        # reload from disk
        tl2 = Timeline(d)
        assert set(tl2.nodes) == set(tl.nodes)
        tree = tl2.to_tree()
        assert len(tree) == 1 and len(tree[0]["children"][0]["children"]) == 2
        assert tl2.wsave_path(root.save_id) == os.path.join(d, "a.wsave")

        # unknown parent rejected
        try:
            tl2.add(path="x.wsave", latent_frame=1, mode="state", parent_id="nope")
            raise AssertionError("unknown parent should be rejected")
        except KeyError:
            pass
    print("PASS fork_tree_and_persistence")


if __name__ == "__main__":
    test_fork_tree_and_persistence()
    print("ALL 1 TESTS PASS")
