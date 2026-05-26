import unittest

from nibi_model_compare.som_missed_creatures import parse_som_response


class ParseSomResponseTests(unittest.TestCase):
    def test_happy_path(self):
        text = '<answer>{"accepted_marks": [1, 3, 5]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 3, 5])

    def test_prose_before_tag(self):
        text = (
            "Looking at the marked image, marks 2 and 4 look like substrate.\n"
            "Marks 1 and 3 are clearly biological.\n"
            '<answer>{"accepted_marks": [1, 3]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 3])

    def test_multiple_answer_blocks_takes_last(self):
        text = (
            '<answer>{"accepted_marks": [1]}</answer>\n'
            "wait, on reflection...\n"
            '<answer>{"accepted_marks": [1, 2]}</answer>'
        )
        self.assertEqual(parse_som_response(text), [1, 2])

    def test_empty_list_is_valid(self):
        text = '<answer>{"accepted_marks": []}</answer>'
        self.assertEqual(parse_som_response(text), [])

    def test_missing_tag_returns_empty(self):
        self.assertEqual(parse_som_response("just text"), [])

    def test_malformed_json_returns_empty(self):
        self.assertEqual(parse_som_response("<answer>{not json}</answer>"), [])

    def test_out_of_range_ids_filtered_when_valid_ids_given(self):
        text = '<answer>{"accepted_marks": [1, 99, 3]}</answer>'
        self.assertEqual(
            parse_som_response(text, valid_ids={1, 2, 3}),
            [1, 3],
        )

    def test_non_int_ids_filtered(self):
        text = '<answer>{"accepted_marks": [1, "2", 3.5, true, 4]}</answer>'
        self.assertEqual(parse_som_response(text), [1, 4])

    def test_nested_json_in_answer_payload(self):
        text = '<answer>{"accepted_marks": [1, 3], "meta": {"note": "ok"}}</answer>'
        self.assertEqual(parse_som_response(text), [1, 3])

    def test_non_string_input_returns_empty(self):
        self.assertEqual(parse_som_response(None), [])
        self.assertEqual(parse_som_response(12345), [])
        self.assertEqual(parse_som_response(""), [])

    def test_empty_valid_ids_filters_everything(self):
        text = '<answer>{"accepted_marks": [1, 2, 3]}</answer>'
        self.assertEqual(parse_som_response(text, valid_ids=set()), [])


from nibi_model_compare.som_missed_creatures import select_target_frames


class SelectTargetFramesTests(unittest.TestCase):
    def _row(self, idx, **kw):
        base = {"frame_index": idx, "num_masks": 3, "error": None, "skipped": False}
        base.update(kw)
        return base

    def test_uniform_spacing_picks_K_valid_frames(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(frame_results, strategy="uniform", k=5)
        self.assertEqual(len(out), 5)
        # uniformly spaced across 0..19 using round(i * step) where step = 19/4 = 4.75
        # -> round(0)=0, round(4.75)=5, round(9.5)=10, round(14.25)=14, round(19)=19
        self.assertEqual(out, [0, 5, 10, 14, 19])

    def test_uniform_skips_errored(self):
        frame_results = [
            self._row(0),
            self._row(1, error="IndexError"),
            self._row(2),
            self._row(3, error="ValueError"),
            self._row(4),
            self._row(5),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid frames are [0, 2, 4, 5]; uniform spacing of 3 over 4 picks
        # indices 0, 2, 3 of that list -> [0, 4, 5]
        self.assertEqual(out, [0, 4, 5])

    def test_uniform_skips_skipped_and_zero_mask(self):
        frame_results = [
            self._row(0),
            self._row(1, skipped=True),
            self._row(2, num_masks=0),
            self._row(3),
            self._row(4),
        ]
        out = select_target_frames(frame_results, strategy="uniform", k=3)
        # valid: [0, 3, 4] -> all picked
        self.assertEqual(out, [0, 3, 4])

    def test_uniform_returns_fewer_when_k_exceeds_valid(self):
        frame_results = [self._row(i, error="x") for i in range(5)]
        frame_results.append(self._row(5))
        out = select_target_frames(frame_results, strategy="uniform", k=10)
        self.assertEqual(out, [5])

    def test_explicit_override_wins(self):
        frame_results = [self._row(i) for i in range(20)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[3, 7, 11]
        )
        self.assertEqual(out, [3, 7, 11])

    def test_explicit_override_filters_invalid_indices(self):
        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="uniform", k=5, explicit=[1, 99, 3]
        )
        self.assertEqual(out, [1, 3])

    def test_motion_strategy_delegates(self):
        called = {}

        def fake_motion(frame_results, k):
            called["ok"] = (len(frame_results), k)
            return [0, 2, 4]

        frame_results = [self._row(i) for i in range(5)]
        out = select_target_frames(
            frame_results, strategy="motion", k=3,
            _motion_selector=fake_motion,
        )
        self.assertEqual(out, [0, 2, 4])
        self.assertEqual(called["ok"], (5, 3))

    def test_include_zero_mask_frames_flag(self):
        frame_results = [
            self._row(0, num_masks=0),
            self._row(1, num_masks=0),
            self._row(2),
        ]
        out = select_target_frames(
            frame_results, strategy="uniform", k=3,
            include_zero_mask_frames=True,
        )
        self.assertEqual(out, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
