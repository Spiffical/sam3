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


if __name__ == "__main__":
    unittest.main()
