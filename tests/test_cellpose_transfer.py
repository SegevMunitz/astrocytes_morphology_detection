"""Tests for safe two-to-three-channel Cellpose transfer initialization."""

import torch

from astroseg.cellpose import (
    expand_cellpose_input_weights,
    prepare_three_channel_cellpose_image,
)


def test_cellpose_transfer_preserves_two_channel_activation_with_zero_gfp() -> None:
    source_weight = torch.randn(4, 2, 3, 3)
    source = {
        "input.weight": source_weight,
        "input_norm.weight": torch.tensor([2.0, 3.0]),
        "output.bias": torch.randn(4),
    }
    target = {
        "input.weight": torch.randn(4, 3, 3, 3),
        "input_norm.weight": torch.ones(3),
        "output.bias": torch.zeros(4),
    }

    converted, expanded_keys = expand_cellpose_input_weights(source, target)
    old_input = torch.randn(2, 2, 12, 13)
    new_input = torch.stack(
        (old_input[:, 0], torch.zeros_like(old_input[:, 0]), old_input[:, 1]), dim=1
    )

    old_output = torch.nn.functional.conv2d(old_input, source_weight)
    new_output = torch.nn.functional.conv2d(new_input, converted["input.weight"])

    assert expanded_keys == ("input.weight",)
    torch.testing.assert_close(new_output, old_output)
    torch.testing.assert_close(converted["output.bias"], source["output.bias"])
    torch.testing.assert_close(
        converted["input_norm.weight"], torch.tensor([2.0, 1.0, 3.0])
    )


def test_cellpose_transfer_rejects_an_unexpected_architecture_change() -> None:
    source = {"weight": torch.ones(4, 2, 3, 3)}
    target = {"weight": torch.ones(5, 3, 3, 3)}

    try:
        expand_cellpose_input_weights(source, target)
    except ValueError as exception:
        assert "Unsupported" in str(exception)
    else:
        raise AssertionError("Unexpected architecture change was accepted")


def test_three_channel_image_conversion_accepts_channel_last_and_first() -> None:
    image = torch.arange(5 * 7 * 3).reshape(5, 7, 3).numpy()

    converted = prepare_three_channel_cellpose_image(image)

    assert converted.shape == (3, 5, 7)
    assert converted.flags.c_contiguous
    torch.testing.assert_close(torch.from_numpy(converted), torch.from_numpy(image).movedim(-1, 0))
    same = prepare_three_channel_cellpose_image(converted)
    torch.testing.assert_close(torch.from_numpy(same), torch.from_numpy(converted))

    four_plane = torch.arange(4 * 5 * 7).reshape(4, 5, 7).numpy()
    fluorescence = prepare_three_channel_cellpose_image(four_plane)
    torch.testing.assert_close(
        torch.from_numpy(fluorescence), torch.from_numpy(four_plane[:3])
    )

    two_plane = torch.arange(2 * 5 * 7).reshape(2, 5, 7).numpy()
    with_missing_gfp = prepare_three_channel_cellpose_image(two_plane)
    torch.testing.assert_close(
        torch.from_numpy(with_missing_gfp[0]), torch.from_numpy(two_plane[0])
    )
    assert not with_missing_gfp[1].any()
    torch.testing.assert_close(
        torch.from_numpy(with_missing_gfp[2]), torch.from_numpy(two_plane[1])
    )
