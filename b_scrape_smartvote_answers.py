import requests
import json


class SmartVoteScraper:
    def __init__(self):
        self.base_url = "https://backend.smartvote.ch/api/graphql"
        self.headers = {"Content-Type": "application/json", "lang": "de"}
        self.election_id = "1057"

    def get_all_candidate_details(self) -> dict:
        """Get all candidates and their complete details in a single query"""
        query = """
        query AllCandidatesWithDetails($searchParams: CandidateSearchParams!) {
          candidates(searchParams: $searchParams) {
            id
            firstname
            lastname
            yearOfBirth
            city
            country
            publicEmailAddress
            gender
            smartmonitorUrl
            profileImageUrl
            isIncumbent
            isElected
            partyAbbreviation
            partyColor
            hasSmartvoteProfile
            website
            facebook
            twitter
            instagram
            video
            additionalLink
            occupation
            employers
            hobbies
            favoriteBooks
            favoriteMovies
            favoriteMusic
            newVideoUrl
            district {
              id
              name
            }
            party {
              id
              color
              name
              abbreviation
            }
            list {
              id
              name
            }
            listPlaces {
              id
              position
              number
            }
            denomination {
              id
              name
            }
            civilState {
              id
              name
            }
            nofChildren
            slogan {
              originalLanguage {
                id
              }
              originalValue
              translatedValue
            }
            commentedTopics {
              id
              name
              comment
              sortorder
            }
            funding {
              id
              amount
              comment {
                originalLanguage {
                  id
                }
                originalValue
                translatedValue
              }
            }
            politicalMandates {
              id
              name
              dateStart
              dateEnd
            }
            vestedInterests {
              id
              sortorder
              name
              board {
                id
                name
              }
              position {
                id
                name
              }
              legalForm {
                id
                name
              }
            }
            education {
              id
              name
            }
            smartspider {
              id
              options {
                id
                cssClass
                fill
              }
              axes {
                id
                cleavage {
                  id
                  name
                }
                value
              }
            }
            answers {
              id
              questionId
              value
              weight
              comment {
                originalLanguage {
                  id
                }
                originalValue
                translatedValue
              }
            }
            isBookmarked
          }
        }
        """

        variables = {
            "searchParams": {
                "electionId": self.election_id,
                "hasSmartvoteProfile": True,
                "isElected": True
            }
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
            return response.json()["data"]["candidates"]
        else:
            raise Exception(f"Query failed with status code: {response.status_code}")

    def save_to_json(self, data: dict, filename: str):
        """Save the data to a JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    scraper = SmartVoteScraper()

    # Get all data in a single request
    candidates = scraper.get_all_candidate_details()

    # Save to JSON file
    scraper.save_to_json(candidates, './data/answers/nationalrat_members.json')
    print(f"Saved information for {len(candidates)} candidates")


if __name__ == "__main__":
    main()