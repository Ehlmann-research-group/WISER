"""A flexible test runner for verifying the order-independence of grouped operations.

This module defines the `TestRunner` class, which allows you to group functions into
sequences for testing, permute the execution order of those functions, and ensure the
output remains consistent regardless of order. This is especially useful for testing
side-effect-free code or systems that should produce the same output regardless of the
order of operations within defined groups.

Example:
    test_runner = TestRunner(None)
    test_runner.add_test_group([partial(func1), partial(func2)])
    test_runner.add_test_group([partial(func3)])
    result = test_runner.run()
"""
from typing import List, Any, Optional
import itertools
from functools import partial


class TestRunner:
    """
    A runner for executing groups of test commands in different orders.

    This class is designed to test the order-independence of certain function sequences.
    You can add multiple groups of functions, where each group is either:
    - A list of one function (executed as-is)
    - A list of multiple functions (executed in all possible permutations)

    The result of each full execution path is compared, and if any permutation yields
    a different result, the runner raises an error.

    Args:
        test_model (WiserTestModel): An object that provides test context (not used directly).

    Attributes:
        test_model: An object that provides test context (not used directly).
        test_groups (List[List[partial]]): A list of function groups to be tested.
    """

    def __init__(self, test_model):
        self.test_model = test_model
        self.test_groups = []

    def add_test_group(self, func_list: List[partial]) -> None:
        """
        Adds a group of functions to be tested.

        If the group has multiple functions, their permutations will be tested
        during execution to verify that order doesn't affect the final result.

        Args:
            func_list (List[partial]): A list of partially applied functions or lambdas.
        """
        self.test_groups.append(func_list)

    def _unroll_test_groups(self) -> Optional[List[List[partial]]]:
        """
        Generates all permutations of function groups and combines them into full test paths.

        Each group with multiple functions will be permuted. Then, all permutations
        across groups are combined to create every possible sequence of operations.

        Returns:
            List[List[partial]]: A list of function call sequences to test.
        """
        unrolled_groups = []

        for group in self.test_groups:
            if len(group) > 1:
                perm_tuples = list(itertools.permutations(group))
                permutations = [list(p) for p in perm_tuples]
            else:
                permutations = [group]

            if len(unrolled_groups) == 0:
                unrolled_groups = permutations
            else:
                new_unrolled_groups = []
                for perm in permutations:
                    for exisiting_group in unrolled_groups:
                        new_unrolled_groups.append(exisiting_group + perm)

                unrolled_groups = new_unrolled_groups

        return unrolled_groups

    def _verify_test_groups(self) -> bool:
        """
        Verifies that the final test group contains exactly one function.

        This ensures a consistent endpoint for comparison across permutations.

        Returns:
            bool: True if the last group contains exactly one function.
        """
        return len(self.test_groups[-1]) == 1

    def _func_group_to_string(functions):
        """
        Converts a list of partial functions into a string representation.

        This helps trace which function order caused a mismatch in results.

        Args:
            functions (List[partial]): A list of partially applied functions.

        Returns:
            str: A string showing the order of function names.
        """
        call_trace = []
        for fn in functions:
            # Retrieve the function's name
            call_trace.append(fn.func.__name__)
        return " -> ".join(call_trace)

    def run_once(self, test_group: List[Any]) -> Any:
        """
        Executes a single sequence of functions and returns the final result.

        Args:
            test_group (List[Callable]): A list of partially applied or lambda functions.

        Returns:
            Any: The result of the final function in the sequence.

        Raises:
            BaseException: If any function in the sequence raises an error.
        """
        print(f"test_group: {test_group}")
        result = None
        try:
            for func in test_group:
                print(f"func to use: {func}")
                result = func()
                print(f"result is: {result}")
            return result
        except BaseException as e:
            raise e

    def run(self):
        """
        Runs all permutations of the function groups and verifies result consistency.

        For each permutation, this method:
        - Executes the sequence of functions
        - Compares the result with prior executions
        - Raises an error if any permutation produces a different result

        Returns:
            Any: The result of the final function in the last test group.

        Raises:
            ValueError: If the last group has more than one function, or results mismatch.
        """
        verified = self._verify_test_groups()
        if not verified:
            raise ValueError(
                "The length of the last item in your test group must be 1. "
                + f"The current length of the item is {len(self.test_groups[-1])}"
            )
        unrolled_test_groups = self._unroll_test_groups()

        previous_test_group = None
        previous_group_result = None
        for test_group in unrolled_test_groups:
            current_result = self.run_once(test_group)
            if current_result != previous_group_result and previous_group_result is not None:
                raise ValueError(
                    "Previous result does not equal current result!"
                    + f"Previous result call trace: {self._func_group_to_string(previous_test_group)}"
                    + f"Current result call trace: {self._func_group_to_string(test_group)}"
                )

            previous_group_result = current_result
            previous_test_group = test_group

        return previous_group_result


"""
This tests out the functionality of the test_runner class. Feel free to play
with it to get an understanding of how it works.
"""


def test1(x, y, change=False):
    if not change:
        x = x + y
        return x


if __name__ == "__main__":
    x = 0
    test_runner = TestRunner(None)

    # func1 = lambda: test1(x, 5)
    # func2 = lambda: test1(x, 4)

    # func3 = lambda: test1(x, 1)

    func1 = partial(test1, x, 5)
    func2 = partial(test1, x, 4)

    func3 = partial(test1, x, 1)

    answer = 10

    first_group = []

    test_runner.add_test_group([func1, func2])
    test_runner.add_test_group([func3])

    result = test_runner.run()

    print(f"result: {result}")
