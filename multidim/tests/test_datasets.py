from multidim.dynaboard_datasets import Review


def test_review_parsing():
    dataset = Review.from_jsonlines('test_fixtures/mini_amazon_reviews.jsonl')
    assert len(dataset) == 3
    assert dataset[0].uid == "1234"
    assert dataset[0].statement == "this is a positive statement"
    assert dataset[0].label == 'positive'

    assert dataset[1].uid == "9999"
    assert dataset[1].statement == "this is a negative statement"
    assert dataset[1].label == 'negative'

    assert dataset[2].uid == "1"
    assert dataset[2].label == 'neutral'
    assert dataset[2].statement == "this is a neutral statement"