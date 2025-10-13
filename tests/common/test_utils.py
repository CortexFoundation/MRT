import pytest
import threading
from mrt.common.utils import N

class TestN:
    def setup_method(self):
        """Reset global N instance before each test method."""
        # Create a new N instance and register it as the global one.
        # This instance will have name="" and counter=0.
        N.register_global(N())

    def test_n_global_scope(self):
        assert N.n() == "%0"
        assert N.n() == "%1"
        assert N.n(prefix="var_") == "var_2"
        assert N.n(suffix="_s") == "%3_s"

    def test_n_scoped_allocation(self):
        # Global scope name allocation
        assert N.n() == "%0"

        with N("scope1"):
            assert N.n() == "scope1_%0"
            assert N.n() == "scope1_%1"
        
        # Back to global scope. Counter of global scope continues.
        assert N.n() == "%1"

    def test_n_nested_scopes(self):
        with N("scope1"):
            assert N.n() == "scope1_%0"
            with N("scope2"):
                assert N.n() == "scope2_%0"
                assert N.n() == "scope2_%1"
            # Back to scope1. Counter of scope1 continues.
            assert N.n() == "scope1_%1"
        
        # Back to global scope
        assert N.n() == "%0" # Global counter was not used yet.

    def test_n_thread_safety(self):
        from queue import Queue
        num_threads = 10
        iterations = 100
        total_names = num_threads * iterations
        results_queue = Queue()
        
        def worker():
            for _ in range(iterations):
                results_queue.put(N.n())

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert results_queue.qsize() == total_names

        generated_names = set()
        while not results_queue.empty():
            generated_names.add(results_queue.get())

        assert len(generated_names) == total_names

        # Check the format and sequence of names
        expected_numbers = set(range(total_names))
        generated_numbers = set()
        for name in generated_names:
            assert name.startswith("%")
            num_str = name[1:]
            assert num_str.isdigit()
            generated_numbers.add(int(num_str))

        assert generated_numbers == expected_numbers

if __name__ == "__main__":
    pytest.main(["-s", __file__, "-vvv"])
