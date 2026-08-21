import numpy as np
import torch
import math
import pytest
from BayesCompare.dist_utils import (
    simplify_string,
    check_array,
    check_small_negative,
    check_small_negative_eigenval,
    check_cos_output,
)
from numpy.testing import assert_allclose
from torch.testing import assert_close


@pytest.mark.parametrize(
    "input_string, expected",
    [  # modify these strings
        ("Hello World", "helloworld"),
        ("Hello_World", "helloworld"),
        ("Hello-World", "helloworld"),
        ("Hello _- World", "helloworld"),
        ("HELLO WORLD", "helloworld"),
        ("Test123", "test123"),
        ("", ""),
        ("already_simple", "alreadysimple"),
        # preserve these strings
        ("Hello.World", "hello.world"),
        ("Hello/World", "hello/world"),
        ("hello", "hello"),
    ],
)
def test_simplfy_string(input_string, expected):
    assert simplify_string(input_string) == expected


def test_check_array():
    # one element
    assert (
        isinstance(check_array(np.array([25.34])), float)
        and check_array(np.array([25.34])) == 25.34
    )
    assert (
        isinstance(check_array(torch.Tensor([25.34])), float)
        and round(check_array(torch.Tensor([25.34])), 2) == 25.34
    )
    # zero dim torch
    assert (
        isinstance(check_array(torch.tensor(25.34)), float)
        and round(check_array(torch.tensor(25.34)), 2) == 25.34
    )
    # more than one element
    with pytest.raises(
        ValueError, match="Distance output is a numpy array with more than one element."
    ):
        check_array(np.array([25.34, 55.36]))
    with pytest.raises(
        ValueError,
        match="Distance output is a torch tensor with more than one element.",
    ):
        check_array(torch.Tensor([25.34, 55.36]))


class TestCheckSmallNegative:

    input_kinds = ["int", "float", "ndarray", "tensor"]

    def _wrap(self, value, kind):
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "ndarray":
            return np.array([value], dtype=np.float64)
        if kind == "tensor":
            return torch.tensor([value], dtype=torch.float64)
        raise ValueError(kind)

    def _to_value(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.item()
        return x

    # -- Core zeroing behavior ------------------------------------------

    @pytest.mark.parametrize("kind", input_kinds)
    def test_small_negative_is_zeroed(self, kind):
        epsilon = 1e-7
        d = (
            self._wrap(-epsilon / 2, kind)
            if kind not in ("int",)
            else self._wrap(0, kind)
        )
        result = check_small_negative(d, epsilon=epsilon)
        assert self._to_value(result) == 0.0

    @pytest.mark.parametrize("kind", ["float", "ndarray", "tensor"])
    def test_negative_larger_than_epsilon_is_unchanged(self, kind):
        epsilon = 1e-7
        d = self._wrap(-1.0, kind)
        result = check_small_negative(d, epsilon=epsilon)
        assert self._to_value(result) == pytest.approx(-1.0)

    @pytest.mark.parametrize("kind", input_kinds)
    def test_positive_value_is_unchanged(self, kind):
        epsilon = 1e-7
        d = self._wrap(5.0, kind)
        result = check_small_negative(d, epsilon=epsilon)
        assert self._to_value(result) == pytest.approx(5.0)

    @pytest.mark.parametrize("kind", input_kinds)
    def test_exact_zero_stays_zero(self, kind):
        d = self._wrap(0.0, kind)
        result = check_small_negative(d)
        assert self._to_value(result) == 0.0

    # -- Epsilon boundary behavior ---------------------------------------

    def test_negative_value_equal_to_epsilon_boundary(self):
        epsilon = 1e-7
        result = check_small_negative(-epsilon, epsilon=epsilon)
        assert result == pytest.approx(-epsilon)

    def test_negative_value_just_inside_epsilon_is_zeroed(self):
        epsilon = 1e-7
        result = check_small_negative(-epsilon * 0.99, epsilon=epsilon)
        assert result == 0.0

    def test_negative_value_just_outside_epsilon_is_unchanged(self):
        epsilon = 1e-7
        result = check_small_negative(-epsilon * 1.01, epsilon=epsilon)
        assert result == pytest.approx(-epsilon * 1.01)

    def test_custom_epsilon_is_respected(self):
        epsilon = 1e-3
        # Would NOT be zeroed under the default epsilon, but should be zeroed
        # under this larger, custom epsilon.
        result = check_small_negative(-5e-4, epsilon=epsilon)
        assert result == 0.0

    def test_default_epsilon_value(self):
        # -1e-8 is smaller in magnitude than the default epsilon (1e-7)
        result = check_small_negative(-1e-8)
        assert result == 0.0

    # -- Type and shape preservation -------------------------------------

    def test_return_type_is_float_for_float_input(self):
        result = check_small_negative(-1e-8, epsilon=1e-7)
        assert isinstance(result, float)

    def test_return_type_is_ndarray_for_ndarray_input(self):
        d = np.array([-1e-8])
        result = check_small_negative(d, epsilon=1e-7)
        assert isinstance(result, np.ndarray)
        assert result.shape == d.shape

    def test_return_type_is_tensor_for_tensor_input(self):
        d = torch.tensor([-1e-8])
        result = check_small_negative(d, epsilon=1e-7)
        assert isinstance(result, torch.Tensor)
        assert result.shape == d.shape

    def test_ndarray_dtype_is_preserved(self):
        d = np.array([-1e-8], dtype=np.float32)
        result = check_small_negative(d, epsilon=1e-7)
        assert result.dtype == np.float32

    def test_tensor_dtype_is_preserved(self):
        d = torch.tensor([-1e-8], dtype=torch.float32)
        result = check_small_negative(d, epsilon=1e-7)
        assert result.dtype == torch.float32

    # -- Numerical edge cases ---------------------------------------------

    def test_nan_input_behavior(self):
        result = check_small_negative(float("nan"), epsilon=1e-7)
        assert math.isnan(result)

    def test_positive_infinity_is_unchanged(self):
        result = check_small_negative(float("inf"), epsilon=1e-7)
        assert result == float("inf")

    def test_negative_infinity_is_unchanged(self):
        result = check_small_negative(float("-inf"), epsilon=1e-7)
        assert result == float("-inf")

    def test_zero_epsilon_never_zeroes_negative_values(self):
        result = check_small_negative(-1e-20, epsilon=0.0)
        assert result == pytest.approx(-1e-20)

    # -- Invalid input types (Raises: TypeError) ---------------------------

    @pytest.mark.parametrize(
        "bad_input",
        [
            "not a number",
            [1.0, 2.0],
            (1.0,),
            {"d": 1.0},
            None,
            1 + 2j,
            {1.0},
        ],
        ids=["str", "list", "tuple", "dict", "none", "complex", "set"],
    )
    def test_raises_type_error_for_unsupported_types(self, bad_input):
        with pytest.raises(TypeError):
            check_small_negative(bad_input, epsilon=1e-7)

    def test_raises_type_error_uses_default_epsilon_too(self):
        with pytest.raises(TypeError):
            check_small_negative("bad")

    def test_numpy_scalar_type_np_float64(self):
        d = np.float64(-1e-8)
        result = check_small_negative(d, epsilon=1e-7)
        assert result == 0.0


class TestCheckSmallNegativeEigenval:

    input_kinds = ["list", "ndarray", "tensor"]

    def _wrap(self, values, kind):
        if kind == "list":
            return list(values)
        if kind == "ndarray":
            return np.array(values, dtype=np.float64)
        if kind == "tensor":
            return torch.tensor(values, dtype=torch.float64)
        raise ValueError(kind)

    def _to_list(self, result):
        if isinstance(result, torch.Tensor):
            return result.tolist()
        if isinstance(result, np.ndarray):
            return result.tolist()
        return list(result)

    # -- Core zeroing behavior, swept across the whole vector -------------

    @pytest.mark.parametrize("kind", input_kinds)
    def test_small_negative_eigenvalues_are_zeroed(self, kind):
        tolerance = 1e-3
        d = self._wrap([-tolerance / 2, 1.0, 2.0], kind)
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 1.0, 2.0])

    @pytest.mark.parametrize("kind", input_kinds)
    def test_positive_eigenvalues_are_unchanged(self, kind):
        tolerance = 1e-3
        d = self._wrap([1.0, 2.0, 3.0], kind)
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("kind", input_kinds)
    def test_exact_zero_eigenvalue_is_unchanged(self, kind):
        tolerance = 1e-3
        d = self._wrap([0.0, 1.0], kind)
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 1.0])

    @pytest.mark.parametrize("kind", input_kinds)
    def test_multiple_small_negative_eigenvalues_all_zeroed(self, kind):
        tolerance = 1e-3
        d = self._wrap([-1e-5, -1e-4, 5.0, -2e-5], kind)
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 0.0, 5.0, 0.0])

    @pytest.mark.parametrize("kind", input_kinds)
    def test_single_element_vector(self, kind):
        tolerance = 1e-3
        d = self._wrap([-1e-5], kind)
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0])

    # -- ValueError contract for large negative eigenvalues ----------------

    @pytest.mark.parametrize("kind", input_kinds)
    def test_large_negative_eigenvalue_raises_value_error(self, kind):
        tolerance = 1e-3
        d = self._wrap([1.0, -1.0, 2.0], kind)  # -1.0 is far below -tolerance
        with pytest.raises(ValueError):
            check_small_negative_eigenval(d, tolerance=tolerance)

    def test_no_error_when_all_eigenvalues_are_within_tolerance_or_positive(self):
        tolerance = 1e-3
        d = [3.0, -1e-4, 0.0, 2.0]
        # Should not raise.
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([3.0, 0.0, 0.0, 2.0])

    # -- Tolerance boundary behavior ---------------------------------------

    def test_eigenvalue_exactly_at_negative_tolerance_boundary(self):
        tolerance = 1e-3
        d = [-tolerance, 1.0]
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 1.0])

    def test_eigenvalue_just_inside_tolerance_is_zeroed(self):
        tolerance = 1e-3
        d = [-tolerance * 0.99, 1.0]
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 1.0])

    def test_eigenvalue_just_outside_tolerance_raises(self):
        tolerance = 1e-3
        d = [-tolerance * 1.01, 1.0]
        with pytest.raises(ValueError):
            check_small_negative_eigenval(d, tolerance=tolerance)

    def test_custom_tolerance_is_respected(self):
        tolerance = 1.0
        d = [-0.5, 2.0]
        result = check_small_negative_eigenval(d, tolerance=tolerance)
        assert self._to_list(result) == pytest.approx([0.0, 2.0])

    def test_default_tolerance_value(self):
        result = check_small_negative_eigenval([-1e-4, 1.0])
        assert self._to_list(result) == pytest.approx([0.0, 1.0])

    def test_zero_tolerance_raises_for_any_negative_value(self):
        with pytest.raises(ValueError):
            check_small_negative_eigenval([-1e-20, 1.0], tolerance=0.0)

    # -- Type and shape preservation ---------------------------------------

    def test_return_type_is_list_for_list_input(self):
        result = check_small_negative_eigenval([-1e-5, 1.0], tolerance=1e-3)
        assert isinstance(result, list)

    def test_return_type_is_ndarray_for_ndarray_input(self):
        d = np.array([-1e-5, 1.0])
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert isinstance(result, np.ndarray)
        assert result.shape == d.shape

    def test_return_type_is_tensor_for_tensor_input(self):
        d = torch.tensor([-1e-5, 1.0])
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert isinstance(result, torch.Tensor)
        assert result.shape == d.shape

    def test_ndarray_dtype_is_preserved(self):
        d = np.array([-1e-5, 1.0], dtype=np.float32)
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert result.dtype == np.float32

    def test_tensor_dtype_is_preserved(self):
        d = torch.tensor([-1e-5, 1.0], dtype=torch.float32)
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert result.dtype == torch.float32

    def test_output_length_matches_input_length(self):
        d = [-1e-5, 1.0, -1e-6, 3.0, 0.0]
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert len(result) == len(d)

    # -- Empty input ----------------------------------------------------------

    @pytest.mark.parametrize("kind", input_kinds)
    def test_empty_input_returns_empty_without_error(self, kind):
        d = self._wrap([], kind)
        result = check_small_negative_eigenval(d, tolerance=1e-3)
        assert len(result) == 0


class TestCheckCosOutput:
    @pytest.mark.parametrize(
        "cos_val, expected",
        [
            (-1.0, -1.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (1.0 + 1e-8, 1.0),  # within epsilon -> clamp
            (1.0 + 5e-8, 1.0),  # within epsilon -> clamp
            (1.0 + 1e-7, 1.0 + 1e-7),  # exactly 1 + epsilon -> unchanged
            (1.0 + 1e-6, 1.0 + 1e-6),  # clearly outside epsilon
        ],
    )
    def test_check_cos_output_scalar(self, cos_val, expected):
        result = check_cos_output(cos_val)
        assert result == pytest.approx(expected)

    def test_check_cos_output_at_epsilon_boundary(self):
        epsilon = 0.01
        result = check_cos_output(1.01, epsilon=epsilon)
        assert result == pytest.approx(1.01)

    def test_check_cos_output_just_inside_epsilon(self):
        epsilon = 0.01
        result = check_cos_output(1.005, epsilon=epsilon)
        assert result == pytest.approx(1.0)

    def test_check_cos_output_numpy_array(self):
        cos_val = [
            -0.5,
            0.0,
            0.8,
            1.0,
            1.0 + 5e-8,
            1.0 + 1e-5,
        ]
        expected = [
            -0.5,
            0.0,
            0.8,
            1.0,
            1.0,
            1.0 + 1e-5,
        ]
        for c_val, exp in zip(cos_val, expected):
            result = check_cos_output(np.array(c_val))
            assert_allclose(result, np.array(exp))
            assert isinstance(result, np.ndarray)
        for c_val, exp in zip(cos_val, expected):
            result = check_cos_output(np.array([c_val]))
            assert_allclose(result, np.array([exp]))
            assert isinstance(result, np.ndarray)

    def test_check_cos_output_tensor(self):
        cos_val = [
            -0.5,
            0.0,
            0.8,
            1.0,
            1.0 + 5e-8,
            1.0 + 1e-5,
        ]
        expected = [
            -0.5,
            0.0,
            0.8,
            1.0,
            1.0,
            1.0 + 1e-5,
        ]
        for c_val, exp in zip(cos_val, expected):
            result = check_cos_output(torch.tensor(c_val, dtype=torch.float64))
            assert_close(result, torch.tensor(exp, dtype=torch.float64))
            assert isinstance(result, torch.Tensor)
        for c_val, exp in zip(cos_val, expected):
            result = check_cos_output(torch.tensor([c_val], dtype=torch.float64))
            assert_close(result, torch.tensor([exp], dtype=torch.float64))
            assert isinstance(result, torch.Tensor)
