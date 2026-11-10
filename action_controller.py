
import subprocess


APP_CYCLE = [
    "Safari",
    "Spotify",
    "Finder",
]


class ActionController:

    def __init__(self):
        self.last_cycle_index = None

    def run_command(self, command):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )

            return result.stdout.strip()

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError
        ) as error:
            stderr = getattr(error, "stderr", None)

            print(
                "Command failed:",
                stderr.strip() if stderr else error
            )

            return None

    def run_applescript(self, script):
        return self.run_command([
            "osascript",
            "-e",
            script
        ])

    def get_frontmost_app(self):
        script = '''
        tell application "System Events"
            return name of first application process whose frontmost is true
        end tell
        '''

        return self.run_applescript(script)

    def open_app(self, app_name):
        print(f"Opening: {app_name}")

        result = self.run_command([
            "open",
            "-a",
            app_name
        ])

        if result is None:
            return False

        if app_name in APP_CYCLE:
            self.last_cycle_index = (
                APP_CYCLE.index(app_name)
            )

        return True

    def spotify_play_pause(self):
        script = '''
        tell application "Spotify"
            playpause
        end tell
        '''

        result = self.run_applescript(script)

        if result is None:
            return False

        print("Spotify play/pause triggered")
        return True

    def cycle_app(self):
        # The stored cycle position is authoritative
        # once the controller has switched an app.
        # This prevents the camera window's focus
        # from resetting the sequence.

        if self.last_cycle_index is None:
            current_app = self.get_frontmost_app()

            if current_app in APP_CYCLE:
                current_index = APP_CYCLE.index(
                    current_app
                )
            else:
                current_index = -1
        else:
            current_index = self.last_cycle_index

        next_index = (
            current_index + 1
        ) % len(APP_CYCLE)

        target_app = APP_CYCLE[next_index]

        print(
            f"Cycle -> {target_app}"
        )

        return self.open_app(target_app)

    def execute_action(self, gesture):
        print(f"\nGesture detected: {gesture}")

        if gesture == "THUMBS_UP":
            return self.open_app("Safari")

        if gesture == "PEACE":
            return self.open_app("Spotify")

        if gesture == "OPEN_PALM":
            return self.spotify_play_pause()

        if gesture == "FIST":
            return self.open_app("Finder")

        if gesture == "ROCK_SIGN":
            return self.cycle_app()

        if gesture == "NEUTRAL":
            return False

        print(f"Unknown gesture: {gesture}")
        return False


# Shared instance for imports and direct tests.
controller = ActionController()


def execute_action(gesture):
    return controller.execute_action(gesture)


if __name__ == "__main__":
    print("AI Gesture Action Controller V2")

    actions = {
        "1": "THUMBS_UP",
        "2": "PEACE",
        "3": "OPEN_PALM",
        "4": "FIST",
        "5": "ROCK_SIGN",
    }

    while True:
        print("\n1 - Open Safari")
        print("2 - Open Spotify")
        print("3 - Spotify Play/Pause")
        print("4 - Open Finder")
        print("5 - Cycle Apps")
        print("Q - Quit")

        choice = input(
            "Select action: "
        ).strip().lower()

        if choice == "q":
            break

        gesture = actions.get(choice)

        if gesture is None:
            print("Invalid selection")
            continue

        print(
            "Result:",
            execute_action(gesture)
        )
