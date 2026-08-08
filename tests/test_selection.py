import unittest

from trajplayer.selection import (
    ChainSelectionError,
    format_chain_selection,
    parse_chain_selection,
)


class ChainSelectionTests(unittest.TestCase):
    def test_accepts_single_chains_lists_and_ranges(self) -> None:
        self.assertEqual(parse_chain_selection("2", 8), (2,))
        self.assertEqual(parse_chain_selection("1, 3,5", 8), (1, 3, 5))
        self.assertEqual(parse_chain_selection("2-5", 8), (2, 3, 4, 5))
        self.assertEqual(
            parse_chain_selection("1,3-5,4,8", 8),
            (1, 3, 4, 5, 8),
        )

    def test_accepts_common_pasted_separators(self) -> None:
        self.assertEqual(parse_chain_selection("1\uff0c3\u20135", 6), (1, 3, 4, 5))
        self.assertEqual(parse_chain_selection("1; 2 4", 6), (1, 2, 4))

    def test_rejects_invalid_or_out_of_range_chains(self) -> None:
        for text in ("", "0", "4-2", "a", "1-", "7"):
            with self.subTest(text=text):
                with self.assertRaises(ChainSelectionError):
                    parse_chain_selection(text, 6)

    def test_formats_consecutive_chains_compactly(self) -> None:
        self.assertEqual(format_chain_selection((1, 3, 4, 5, 8)), "1,3-5,8")


if __name__ == "__main__":
    unittest.main()
