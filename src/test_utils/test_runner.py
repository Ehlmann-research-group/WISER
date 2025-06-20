from typing import List, Any, Optional
import itertools
from functools import partial


class TestRunner:
    """
    Used to run a sequence of commands that together encompass a test. Allows you to run commands in
    different orders to ensure that order doesn't matter for some groups of operations. Note that
    this does increase the time it takes to run those commands exponentially.
    """

    def __init__(self, test_model):
        self.test_model = test_model
        self.test_groups = []

    def add_test_group(self, func_list: List[partial]) -> None:
        """
        Adds a list of tests to the classes complete list of tests to be run.
        Multiple calls to this function will run the tests in the order they
        were added, but if func_list has more than one element, then those elements
        in func_list will be permuted so they are run in all possible orders.

        Arguments:
        - func_list: A list of partial functions that will be called when this classes
                        run method is called
        """
        self.test_groups.append(func_list)

    def _unroll_test_groups(self) -> Optional[List[List[partial]]]:
        """
        The variable test_groups is a list of lists. In the inner lists are functions
        that, when called, the order shouldnt affect execution. Thus, this function
        permutates those inner lists of functions for all the inner lists and creates
        separates lists of functions. The end result is another list of lists where the
        inner lists contain a series of functions that can be called to get the final result.
        Each inner list should output the same final result when each function in it is called
        sequentially.
        """
        # Have a return list called unrolled_groups
        unrolled_groups = []
        # Loop through self.test_groups (the list of lists), inner list var is called group

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

        # If len(group) > 1, we get a list of permutations of the elements in group
        # which we call permutations

        # We then add each list in permutations to the end of each list in unrolled_groups.
        # This should increase the length of unrolled_groups by len(permutations)
        # If unrolled_groups is empty we simply just put permutations in unrolled_groups.

        # We then returned unrolled_groups

    def _verify_test_groups(self) -> bool:
        """
        Ensures that the test groups list ends with a single list item
        """
        return len(self.test_groups[-1]) == 1

    def _func_group_to_string(functions):
        """
        Takes a list of partial objects, calls each one, and returns
        a string with the names of the functions in the order they were called.

        Written by an LLM, then modified
        """
        call_trace = []
        for fn in functions:
            # Retrieve the function's name
            call_trace.append(fn.func.__name__)
        return " -> ".join(call_trace)

    def run_once(self, test_group: List[Any]) -> Any:
        """
        Takes in a list of partial functions to run (either python partials or lambas).
        Runs these functions in the order they are in the list and returns the output of
        the final partial function in this list

        Arguments
            - test_group: A list of partial functios or lambda functions
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
        verified = self._verify_test_groups()
        if not verified:
            raise ValueError(
                "The length of the last item in your test group must be 1. "
                + f"The current length of the item is {len(self.test_groups[-1])}"
            )
        # Should first get the unrolled groups
        unrolled_test_groups = self._unroll_test_groups()

        # Previous test_group
        previous_test_group = None
        # Previous returned answer
        previous_group_result = None
        # Loop through each of the unrolled test groups
        for test_group in unrolled_test_groups:
            # self.test_model.reset()
            current_result = self.run_once(test_group)
            if (
                current_result != previous_group_result
                and previous_group_result is not None
            ):
                raise ValueError(
                    "Previous result does not equal current result!"
                    + f"Previous result call trace: {self._func_group_to_string(previous_test_group)}"
                    + f"Current result call trace: {self._func_group_to_string(test_group)}"
                )

            previous_group_result = current_result
            previous_test_group = test_group

        return previous_group_result

    # TestRunner can take in partials or lambdas (solves problem of parameters)

    # We call run. Run should go through the list of lists The first layer of the lists
    # are things that can run sequentially. THe second layer are things that can be permuted
    # The last item in the lists of lists should always be a one element list and the return
    # value of that function will be used as the output value of the test run

    # So the best way to go about this would be to process the list of lists into multiple
    # lists of things that run sequentially, then iterate through those lists, running
    # the functions and resetting test model after each iteration. We keep the most previous
    # result of all of the previous finction calls.


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
