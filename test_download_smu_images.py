import unittest

from download_smu_images import (
    SearchSpec,
    image_api_url,
    parse_search_url,
    safe_slug,
    search_api_url,
)
from classify_smu_images import (
    build_classification,
    classification_config_id,
    people_bucket,
)


class DownloaderTests(unittest.TestCase):
    def test_parses_supplied_search_url(self):
        spec = parse_search_url(
            "https://digitalcollections.smu.edu/digital/collection/eaa/"
            "search/searchterm/Ag2002.1407/page/1"
        )
        self.assertEqual(spec.base_url, "https://digitalcollections.smu.edu")
        self.assertEqual(spec.collection, "eaa")
        self.assertEqual(spec.search_term, "Ag2002.1407")
        self.assertEqual(spec.field, "all")

    def test_builds_contentdm_and_iiif_urls(self):
        spec = SearchSpec("https://example.test", "eaa", "one two", "title")
        self.assertIn("searchterm/one%20two/field/title", search_api_url(spec, 2))
        self.assertEqual(
            image_api_url(spec, "708", 1600),
            "https://example.test/iiif/2/eaa:708/full/1600,/0/default.jpg",
        )
        self.assertEqual(
            image_api_url(spec, "708", None),
            "https://example.test/iiif/2/eaa:708/full/full/0/default.jpg",
        )

    def test_safe_slug(self):
        self.assertEqual(safe_slug("Nagar Brahmins"), "Nagar_Brahmins")
        self.assertEqual(safe_slug("Crème / photo?"), "Creme_photo")

    def test_people_buckets(self):
        self.assertEqual(people_bucket(0), "no_people")
        self.assertEqual(people_bucket(0, True), "unknown_people_count")
        self.assertEqual(people_bucket(1), "one_person")
        self.assertEqual(people_bucket(2), "two_people")
        self.assertEqual(people_bucket(4), "group_3_to_5")
        self.assertEqual(people_bucket(9), "group_6_plus")

    def test_builds_reviewable_classification(self):
        config_id = classification_config_id("clip", "detector", 0.6, {"people": "p", "landscape": "l"})
        result = build_classification(
            filename="1.jpg",
            scores={"people": 0.51, "landscape": 0.49},
            person_scores=[],
            image_size=(1600, 2000),
            scene_model="clip",
            person_model="detector",
            person_threshold=0.6,
            config_id=config_id,
        )
        self.assertEqual(result.primary_category, "people")
        self.assertEqual(result.people_bucket, "unknown_people_count")
        self.assertTrue(result.needs_review)
        self.assertIn("category_scores_close", result.review_reason)

    def test_detected_people_are_primary_with_separate_scene(self):
        result = build_classification(
            filename="2.jpg",
            scores={"people": 0.12, "architecture_monument": 0.55, "landscape": 0.33},
            person_scores=[0.91, 0.75, 0.62],
            image_size=(1600, 2100),
            scene_model="clip",
            person_model="detector",
            person_threshold=0.4,
            config_id="test",
        )
        self.assertEqual(result.primary_category, "people")
        self.assertEqual(result.scene_category, "architecture_monument")
        self.assertEqual(result.people_bucket, "group_3_to_5")


if __name__ == "__main__":
    unittest.main()
