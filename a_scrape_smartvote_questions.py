import requests
import json


class SmartVoteQuestionnaireScraper:
    def __init__(self):
        self.base_url = "https://backend.smartvote.ch/api/graphql"
        self.headers = {"Content-Type": "application/json", "lang": "de"}
        self.election_id = "1057"

    def get_questionnaire(self) -> dict:
        """Get the complete questionnaire with all questions and categories"""
        query = """
        query QuestionnaireQuery($electionId: ID!) {
          election(id: $electionId) {
            id
            questionnaire {
              id
              hasRapide
              nofQuestions
              categories {
                id
                categoryId
                name
                type
                description
                sortorder
                questions {
                  id
                  type
                  text
                  info
                  pro
                  contra
                  isInRapide
                  sortorder
                  glossaryItems {
                    id
                    term
                    definition
                    startCharacter
                    endCharacter
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "electionId": self.election_id
        }

        response = requests.post(
            self.base_url,
            headers=self.headers,
            json={
                "query": query,
                "variables": variables
            }
        )

        if response.status_code == 200:
            return response.json()["data"]["election"]["questionnaire"]
        else:
            raise Exception(f"Query failed with status code: {response.status_code}")

    def save_to_json(self, data: dict, filename: str):
        """Save the data to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    scraper = SmartVoteQuestionnaireScraper()

    # Get questionnaire data
    questionnaire = scraper.get_questionnaire()

    # Save to JSON file
    scraper.save_to_json(questionnaire, './data/questionnaire/questionnaire.json')

    # Print some statistics
    print(f"Saved questionnaire with {questionnaire['nofQuestions']} questions")
    print(f"Number of categories: {len(questionnaire['categories'])}")

    # Print category names
    print("\nCategories:")
    for category in questionnaire['categories']:
        print(f"- {category['name']} ({len(category['questions'])} questions)")


if __name__ == "__main__":
    main()