"""GraphQL queries and fragments for JobStreet scraper."""

# GraphQL fragment for job details (used in batch queries)
GRAPHQL_FRAGMENT = """
fragment job on JobDetails {
    job {
        sourceZone
        tracking {
            adProductType
            classificationInfo {
                classificationId
                classification
                subClassificationId
                subClassification
                __typename
            }
            hasRoleRequirements
            isPrivateAdvertiser
            locationInfo {
                area
                location
                locationIds
                __typename
            }
            workTypeIds
            postedTime
            __typename
        }
        id
        title
        phoneNumber
        isExpired
        expiresAt {
            dateTimeUtc
            __typename
        }
        isLinkOut
        contactMatches {
            type
            value
            __typename
        }
        isVerified
        abstract
        content(platform: WEB)
        status
        listedAt {
            label(context: JOB_POSTED, length: SHORT, timezone: $timezone, locale: $locale)
            dateTimeUtc
            __typename
        }
        salary {
            currencyLabel(zone: $zone)
            label
            __typename
        }
        shareLink(platform: WEB, zone: $zone, locale: $locale)
        workTypes {
            label(locale: $locale)
            __typename
        }
        advertiser {
            id
            name(locale: $locale)
            isVerified
            registrationDate {
                dateTimeUtc
                __typename
            }
            __typename
        }
        location {
            label(locale: $locale, type: LONG)
            __typename
        }
        classifications {
            label(languageCode: $languageCode)
            __typename
        }
        products {
            branding {
                id
                cover {
                    url
                    __typename
                }
                thumbnailCover: cover(isThumbnail: true) {
                    url
                    __typename
                }
                logo {
                    url
                    __typename
                }
                __typename
            }
            bullets
            questionnaire {
                questions
                __typename
            }
            video {
                url
                position
                __typename
            }
            displayTags {
                label(locale: $locale)
                __typename
            }
            __typename
        }
        __typename
    }
    companyProfile(zone: $zone) {
        id
        name
        companyNameSlug
        shouldDisplayReviews
        branding {
            logo
            __typename
        }
        overview {
            description {
                paragraphs
                __typename
            }
            industry
            size {
                description
                __typename
            }
            website {
                url
                __typename
            }
            __typename
        }
        reviewsSummary {
            overallRating {
                numberOfReviews {
                    value
                    __typename
                }
                value
                __typename
            }
            __typename
        }
        perksAndBenefits {
            title
            __typename
        }
        __typename
    }
    companySearchUrl(zone: $zone, languageCode: $languageCode)
    companyTags {
        key(languageCode: $languageCode)
        value
        __typename
    }
    restrictedApplication(countryCode: $countryCode) {
        label(locale: $locale)
        __typename
    }
    sourcr {
        image
        imageMobile
        link
        __typename
    }
    __typename
}
"""
