import unittest

from utils.auth import AuthConfigError, credentials_match, load_auth_config


class AuthConfigTests(unittest.TestCase):
    def test_exact_credentials_with_slash_and_underscores_are_accepted(self) -> None:
        config = load_auth_config(
            {
                "auth": {
                    "username": "CAFC_Analysts",
                    "password": "Example_Addicks/26_27!",
                }
            }
        )

        self.assertTrue(
            credentials_match(config, "CAFC_Analysts", "Example_Addicks/26_27!")
        )

    def test_username_case_and_outer_whitespace_are_forgiving(self) -> None:
        config = load_auth_config(
            {"auth": {"username": "CAFC_Analysts", "password": "Example-password"}}
        )

        self.assertTrue(credentials_match(config, "  cafc_analysts  ", "Example-password"))

    def test_wrong_password_is_rejected(self) -> None:
        config = load_auth_config(
            {"auth": {"username": "CAFC_Analysts", "password": "Example-password"}}
        )

        self.assertFalse(credentials_match(config, "CAFC_Analysts", "Wrong-password"))

    def test_password_quotes_are_content_not_toml_syntax(self) -> None:
        config = load_auth_config(
            {"auth": {"username": "CAFC_Analysts", "password": '"Example-password"'}}
        )

        self.assertTrue(credentials_match(config, "CAFC_Analysts", '"Example-password"'))
        self.assertFalse(credentials_match(config, "CAFC_Analysts", "Example-password"))

    def test_missing_auth_section_has_a_safe_actionable_error(self) -> None:
        with self.assertRaisesRegex(AuthConfigError, r"missing \[auth\] section"):
            load_auth_config({})

    def test_missing_password_has_a_safe_actionable_error(self) -> None:
        with self.assertRaisesRegex(AuthConfigError, r"missing auth.password"):
            load_auth_config({"auth": {"username": "CAFC_Analysts"}})

    def test_password_rotation_changes_the_session_revision(self) -> None:
        first = load_auth_config(
            {"auth": {"username": "CAFC_Analysts", "password": "First-password"}}
        )
        second = load_auth_config(
            {"auth": {"username": "CAFC_Analysts", "password": "Second-password"}}
        )

        self.assertNotEqual(first.revision, second.revision)


if __name__ == "__main__":
    unittest.main()
