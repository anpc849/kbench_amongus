class NotebookMapUI:
    def __init__(
        self,
        room_coords,
        map_image_path=None,
        width=10,
        height=7,
        show_logs=True,
        pause_seconds=0.2,
    ):
        self.room_coords = room_coords
        self.map_image_path = map_image_path
        self.width = width
        self.height = height
        self.show_logs = show_logs
        self.pause_seconds = pause_seconds
        self.report_text = None
        self.last_env = None
        self.map_handle = None
        self.log_handle = None

    def reset(self):
        self.report_text = None
        self.last_env = None
        self.map_handle = None
        self.log_handle = None

    def draw_map(self, env):
        try:
            from IPython.display import Markdown, display
            import time
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            from matplotlib.patches import Circle, Polygon
        except ImportError:
            return

        self.last_env = env
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.set_xlim(0, 1000)
        ax.set_ylim(560, 0)
        ax.set_aspect("equal")
        ax.axis("off")

        has_background = False
        if self.map_image_path:
            try:
                ax.imshow(mpimg.imread(self.map_image_path), extent=[0, 1000, 560, 0])
                has_background = True
            except FileNotFoundError:
                has_background = False

        for room, roominfo in self.room_coords.items():
            coords = roominfo["coords"]
            xs = coords[0::2]
            ys = coords[1::2]
            if not has_background:
                points = list(zip(xs, ys))
                ax.add_patch(
                    Polygon(points, closed=True, facecolor="white", edgecolor="black")
                )
                label = self._short_room_name(room)
                ax.text(
                    (min(xs) + max(xs)) / 2,
                    min(ys) + 14,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            players = env.map.get_players_in_room(room, include_new_deaths=True)
            x = min(xs) + 12
            y = min(ys) + (max(ys) - min(ys)) / 2
            for player in players:
                ax.add_patch(Circle((x, y), 7, facecolor=player.color, edgecolor="black"))
                if not player.is_alive:
                    ax.plot([x - 7, x + 7], [y - 7, y + 7], color="red", linewidth=2)
                x += 20
                if x > max(xs) - 10:
                    x = min(xs) + 12
                    y += 18

        progress = env.task_assignment.check_task_completion()
        ax.text(
            20,
            545,
            f"Step {env.timestep} | {env.current_phase} | Task Progress {progress * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
        )
        if self.report_text:
            ax.text(
                500,
                280,
                self.report_text,
                ha="center",
                va="center",
                fontsize=12,
                bbox={"facecolor": "white", "edgecolor": "black"},
            )

        if self.map_handle is None:
            self.map_handle = display(fig, display_id=True)
        else:
            self.map_handle.update(fig)
        if self.pause_seconds:
            time.sleep(self.pause_seconds)
        plt.close(fig)

        if self.show_logs:
            rendered_log = Markdown(self._render_log(env))
            if self.log_handle is None:
                self.log_handle = display(rendered_log, display_id=True)
            else:
                self.log_handle.update(rendered_log)

    def report(self, text):
        self.report_text = text
        if self.last_env is not None:
            self.draw_map(self.last_env)
            return
        try:
            from IPython.display import Markdown, display
        except ImportError:
            print(text)
            return
        display(Markdown(f"**{text}**"))

    def quit_UI(self):
        return

    @staticmethod
    def _short_room_name(room):
        return {
            "Upper Engine": "Upper Eng.",
            "Lower Engine": "Lower Eng.",
            "Communications": "Comms",
            "Navigation": "Nav",
        }.get(room, room)

    @staticmethod
    def _activity_text(activity):
        return (
            f"Step {activity['timestep']}: {activity['phase']} phase - "
            f"{activity['player']} {activity['action']}"
        )

    @staticmethod
    def _decision_text(decision):
        thought = decision.get("thought", "").replace("\n", " ").strip()
        if len(thought) > 240:
            thought = thought[:237] + "..."
        return (
            f"Step {decision.get('timestep')}: {decision.get('player')} "
            f"thought: {thought} | action: {decision.get('chosen_action')}"
        )

    def _render_log(self, env):
        lines = ["### Game Log", ""]
        for activity in env.activity_log:
            lines.append(f"- {self._activity_text(activity)}")
        if getattr(env, "decision_log", None):
            lines.extend(["", "### LLM Thoughts", ""])
            for decision in env.decision_log:
                lines.append(f"- {self._decision_text(decision)}")
        return "\n".join(lines)
