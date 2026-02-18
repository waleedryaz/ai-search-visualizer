import tkinter as tk
from tkinter import ttk, messagebox
import collections
import queue
import time

GRID_SIZE  = 20
CELL_SIZE  = 32

COLOR_EMPTY         = "white"
COLOR_WALL          = "#2d2d2d"
COLOR_START         = "#00c853"
COLOR_GOAL          = "#d50000"
COLOR_EXPLORED_S    = "#90caf9"   
COLOR_EXPLORED_G    = "#f48fb1"   
COLOR_PATH          = "#FFD600"
COLOR_MEETING       = "#aa00ff"   

# Data Structures

class Node:
    def __init__(self, row, col, parent=None, cost=0, depth=0):
        self.row    = row
        self.col    = col
        self.parent = parent
        self.cost   = cost
        self.depth  = depth

    def get_pos(self):
        return (self.row, self.col)

    def __lt__(self, other):
        return self.cost < other.cost

# Search Algorithms

class SearchAlgorithms:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def get_neighbors(self, node, walls):
        r, c = node.row, node.col
        for nr, nc in [
            (r-1, c),   (r, c+1),
            (r+1, c),   (r+1, c+1),
            (r, c-1),   (r-1, c-1),
        ]:
            if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in walls:
                yield Node(nr, nc, parent=node, cost=node.cost+1, depth=node.depth+1)

    #  BFS 
    def bfs(self, start, goal, walls):
        sn = Node(*start)
        frontier = collections.deque([sn])
        explored = {start}
        step = 0
        while frontier:
            cur = frontier.popleft()
            step += 1
            yield cur, explored, step, None
            if cur.get_pos() == goal:
                return cur
            for nb in self.get_neighbors(cur, walls):
                if nb.get_pos() not in explored:
                    explored.add(nb.get_pos())
                    frontier.append(nb)

    # DFS 
    def dfs(self, start, goal, walls):
        stack    = [Node(*start)]
        explored = set()
        step = 0
        while stack:
            cur = stack.pop()
            if cur.get_pos() in explored:
                continue
            explored.add(cur.get_pos())
            step += 1
            yield cur, explored, step, None
            if cur.get_pos() == goal:
                return cur
            for nb in reversed(list(self.get_neighbors(cur, walls))):
                if nb.get_pos() not in explored:
                    stack.append(nb)

    #  UCS 
    def ucs(self, start, goal, walls):
        pq = queue.PriorityQueue()
        pq.put(Node(*start, cost=0))
        explored   = set()
        min_costs  = {start: 0}
        step = 0
        while not pq.empty():
            cur = pq.get()
            if cur.get_pos() == goal:
                step += 1
                yield cur, explored, step, None
                return cur
            if cur.get_pos() in explored:
                continue
            explored.add(cur.get_pos())
            step += 1
            yield cur, explored, step, None
            for nb in self.get_neighbors(cur, walls):
                if nb.get_pos() not in min_costs or nb.cost < min_costs[nb.get_pos()]:
                    min_costs[nb.get_pos()] = nb.cost
                    pq.put(nb)

    # DLS 
    def dls(self, start, goal, walls, limit):
        stack    = [Node(*start, depth=0)]
        exp_depth = {}
        step = 0
        while stack:
            cur = stack.pop()
            if cur.get_pos() in exp_depth and exp_depth[cur.get_pos()] <= cur.depth:
                continue
            exp_depth[cur.get_pos()] = cur.depth
            step += 1
            yield cur, set(exp_depth.keys()), step, f"depth {cur.depth}/{limit}"
            if cur.get_pos() == goal:
                return cur
            if cur.depth < limit:
                for nb in reversed(list(self.get_neighbors(cur, walls))):
                    stack.append(nb)

    # IDDFS
    def iddfs(self, start, goal, walls):
        for limit in range(self.rows * self.cols):
            for item in self.dls(start, goal, walls, limit):
                yield item
                if item[0].get_pos() == goal:
                    return item[0]

    # Bidirectional BFS
    def bidirectional(self, start, goal, walls):
        f_s = collections.deque([Node(*start)])
        f_g = collections.deque([Node(*goal)])
        ex_s = {start: Node(*start)}
        ex_g = {goal:  Node(*goal)}
        step = 0

        while f_s or f_g:
            if f_s:
                cs = f_s.popleft()
                step += 1
                yield cs, (set(ex_s), set(ex_g)), step, "S"
                if cs.get_pos() in ex_g:
                    m = self._merge(cs, ex_g[cs.get_pos()])
                    yield m, (set(ex_s), set(ex_g)), step, "MEET"
                    return m
                for nb in self.get_neighbors(cs, walls):
                    if nb.get_pos() not in ex_s:
                        ex_s[nb.get_pos()] = nb
                        f_s.append(nb)

            if f_g:
                cg = f_g.popleft()
                step += 1
                yield cg, (set(ex_s), set(ex_g)), step, "G"
                if cg.get_pos() in ex_s:
                    m = self._merge(ex_s[cg.get_pos()], cg)
                    yield m, (set(ex_s), set(ex_g)), step, "MEET"
                    return m
                for nb in self.get_neighbors(cg, walls):
                    if nb.get_pos() not in ex_g:
                        ex_g[nb.get_pos()] = nb
                        f_g.append(nb)

    def _merge(self, ns, ng):
        ns.special_bidirectional_match = ng
        ns.is_bidirectional_meeting_point = True
        return ns

# GUI

class PathfinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Search Visualizer — Enhanced")
        self.root.resizable(False, False)
        self.root.configure(bg="#eceff1")

        self.canvas = tk.Canvas(
            root,
            width=GRID_SIZE * CELL_SIZE,
            height=GRID_SIZE * CELL_SIZE,
            bg="white",
            highlightthickness=2,
            highlightbackground="#90a4ae"
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.panel = tk.Frame(root, width=265, bg="#eceff1")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=10)
        self.panel.pack_propagate(False)

        self.walls      = set()
        self.start_pos  = None
        self.goal_pos   = None
        self.running    = False
        self.show_nums  = tk.BooleanVar(value=True)
        self.cell_items = {}   # "r_c" -> (rect_id, text_id)

        self._build_panel()
        self._draw_grid()

        self.canvas.bind("<Button-1>",  self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        self.engine = SearchAlgorithms(GRID_SIZE, GRID_SIZE)

    # PANEL

    def _build_panel(self):
        p = self.panel

        tk.Label(p, text="AI Search Visualizer", font=("Segoe UI", 13, "bold"),
                 bg="#eceff1", fg="#1a237e").pack(pady=(8, 0))
        tk.Label(p, text="Artificial Intelligence — Assignment", font=("Segoe UI", 8),
                 bg="#eceff1", fg="#78909c").pack(pady=(0, 4))
        ttk.Separator(p).pack(fill="x", pady=4)

        # Instructions
        tk.Label(p,
            text="① 1st click  → Place Start (green)\n"
                 "② 2nd click → Place Goal  (red)\n"
                 "③ Click/Drag → Draw Walls\n"
                 "④ Pick algorithm & press Start",
            font=("Segoe UI", 8), bg="#eceff1", justify=tk.LEFT
        ).pack(anchor="w", padx=8, pady=2)
        ttk.Separator(p).pack(fill="x", pady=4)

        # Algorithm
        tk.Label(p, text="Algorithm:", font=("Segoe UI", 9, "bold"), bg="#eceff1").pack(anchor="w", padx=8)
        self.algo_var = tk.StringVar(value="BFS")
        cb = ttk.Combobox(p, textvariable=self.algo_var, state="readonly",
                          values=("BFS","DFS","UCS","DLS","IDDFS","Bidirectional"))
        cb.pack(fill="x", padx=8, pady=3)
        cb.bind("<<ComboboxSelected>>", self._on_algo_change)

        # DLS frame
        self.dls_frame = tk.Frame(p, bg="#eceff1")
        tk.Label(self.dls_frame, text="Depth Limit:", font=("Segoe UI", 9),
                 bg="#eceff1").pack(side=tk.LEFT, padx=(8,4))
        self.dls_var = tk.StringVar(value="15")
        tk.Spinbox(self.dls_frame, from_=1, to=400, textvariable=self.dls_var,
                   width=7, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        # hidden initially
        self.dls_frame.pack(fill="x", padx=0, pady=2)
        self.dls_frame.pack_forget()

        # Speed
        ttk.Separator(p).pack(fill="x", pady=4)
        tk.Label(p, text="Animation Speed:", font=("Segoe UI", 9, "bold"), bg="#eceff1").pack(anchor="w", padx=8)
        sf = tk.Frame(p, bg="#eceff1")
        sf.pack(fill="x", padx=8)
        tk.Label(sf, text="Fast", bg="#eceff1", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.02)
        tk.Scale(sf, from_=0.001, to=0.15, resolution=0.001,
                 orient=tk.HORIZONTAL, variable=self.speed_var,
                 bg="#eceff1", showvalue=False, length=130).pack(side=tk.LEFT)
        tk.Label(sf, text="Slow", bg="#eceff1", font=("Segoe UI", 8)).pack(side=tk.LEFT)

        # Show cell numbers
        tk.Checkbutton(p, text="Show Cell Numbers", variable=self.show_nums,
                       command=self._toggle_numbers,
                       bg="#eceff1", font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=3)

        ttk.Separator(p).pack(fill="x", pady=4)

        # Buttons
        def btn(text, cmd, bg, fg="white"):
            tk.Button(p, text=text, command=cmd, bg=bg, fg=fg,
                      font=("Segoe UI", 9, "bold"), relief="flat",
                      padx=4, pady=5, cursor="hand2").pack(fill="x", padx=8, pady=2)

        btn("▶  Start Search",  self.run_search,  "#2e7d32")
        btn("↺  Reset Grid",    self.reset_grid,   "#1565c0")
        btn("⊘  Clear Walls",   self.clear_walls,  "#6d4c41")

        ttk.Separator(p).pack(fill="x", pady=4)

        # Status / stats
        tk.Label(p, text="Status:", font=("Segoe UI", 9, "bold"), bg="#eceff1").pack(anchor="w", padx=8)
        self.status_var = tk.StringVar(value="Ready — place Start & Goal")
        tk.Label(p, textvariable=self.status_var, fg="#1a237e", bg="#eceff1",
                 wraplength=245, justify=tk.LEFT, font=("Segoe UI", 9)).pack(anchor="w", padx=8)

        self.stats_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self.stats_var, fg="#4a148c", bg="#eceff1",
                 wraplength=245, justify=tk.LEFT, font=("Segoe UI", 8)).pack(anchor="w", padx=8, pady=2)

        ttk.Separator(p).pack(fill="x", pady=4)

        # Legend
        tk.Label(p, text="Legend:", font=("Segoe UI", 9, "bold"), bg="#eceff1").pack(anchor="w", padx=8)
        for clr, lbl in [
            (COLOR_START,      "Start"),
            (COLOR_GOAL,       "Goal"),
            (COLOR_WALL,       "Wall"),
            (COLOR_EXPLORED_S, "Explored — Start side"),
            (COLOR_EXPLORED_G, "Explored — Goal side"),
            (COLOR_PATH,       "Final Path"),
            (COLOR_MEETING,    "Meeting Point (Bidir.)"),
        ]:
            row = tk.Frame(p, bg="#eceff1")
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, bg=clr, width=2, relief="solid", bd=1).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {lbl}", bg="#eceff1", font=("Segoe UI", 8)).pack(side=tk.LEFT)

    def _on_algo_change(self, _=None):
        if self.algo_var.get() == "DLS":
            self.dls_frame.pack(fill="x", padx=0, pady=2)
        else:
            self.dls_frame.pack_forget()

    def _toggle_numbers(self):
        st = "normal" if self.show_nums.get() else "hidden"
        for _, (_, tid) in self.cell_items.items():
            if tid:
                self.canvas.itemconfig(tid, state=st)
    # GRID

    def _draw_grid(self):
        self.canvas.delete("all")
        self.cell_items = {}
        num = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                num += 1
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                cx, cy  = (x1+x2)/2, (y1+y2)/2
                color   = self._cell_color(r, c)

                rid = self.canvas.create_rectangle(x1, y1, x2, y2,
                    fill=color, outline="#cfd8dc", tags=f"cell_{r}_{c}")

                tc = "#cccccc" if color == "white" else "#444444"
                tid = self.canvas.create_text(cx, cy, text=str(num),
                    font=("Arial", 7), fill=tc,
                    state="normal" if self.show_nums.get() else "hidden",
                    tags=f"num_{r}_{c}")

                self.cell_items[f"{r}_{c}"] = (rid, tid)

    def _cell_color(self, r, c):
        if (r, c) == self.start_pos: return COLOR_START
        if (r, c) == self.goal_pos:  return COLOR_GOAL
        if (r, c) in self.walls:     return COLOR_WALL
        return "white"

    def _set_cell(self, r, c, color):
        # Preserve special cells
        if   (r, c) == self.start_pos: color = COLOR_START
        elif (r, c) == self.goal_pos:  color = COLOR_GOAL
        elif (r, c) in self.walls:     color = COLOR_WALL

        key = f"{r}_{c}"
        if key not in self.cell_items: return
        rid, tid = self.cell_items[key]
        self.canvas.itemconfig(rid, fill=color)
        # text contrast
        dark = color in (COLOR_WALL,)
        light_bg = color in ("white", COLOR_EXPLORED_S, COLOR_EXPLORED_G)
        tc = "#888888" if light_bg else ("#ffffff" if dark else "#333333")
        if tid:
            self.canvas.itemconfig(tid, fill=tc)

    # MOUSE

    def on_click(self, e):
        if self.running: return
        r, c = e.y // CELL_SIZE, e.x // CELL_SIZE
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE): return
        if self.start_pos is None:
            self.start_pos = (r, c)
            self.walls.discard((r, c))
        elif self.goal_pos is None and (r, c) != self.start_pos:
            self.goal_pos = (r, c)
            self.walls.discard((r, c))
        elif (r, c) not in (self.start_pos, self.goal_pos):
            if (r, c) in self.walls: self.walls.remove((r, c))
            else:                     self.walls.add((r, c))
        self._draw_grid()

    def on_drag(self, e):
        if self.running: return
        r, c = e.y // CELL_SIZE, e.x // CELL_SIZE
        if not (0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE): return
        if (r, c) not in (self.start_pos, self.goal_pos):
            self.walls.add((r, c))
            self._set_cell(r, c, COLOR_WALL)

    # CONTROLS

    def reset_grid(self):
        self.running   = False
        self.start_pos = None
        self.goal_pos  = None
        self.walls     = set()
        self._draw_grid()
        self.status_var.set("Ready — place Start & Goal")
        self.stats_var.set("")

    def clear_walls(self):
        self.running = False
        self.walls   = set()
        self._draw_grid()

    # RUN SEARCH

    def run_search(self):
        if self.running: return
        if not self.start_pos or not self.goal_pos:
            messagebox.showwarning("Missing Points", "Place Start and Goal first.")
            return

        algo = self.algo_var.get()

        # DLS limit
        if algo == "DLS":
            try:
                limit = int(self.dls_var.get())
                if limit < 1: raise ValueError
            except ValueError:
                messagebox.showerror("Bad Limit", "Depth Limit must be a positive integer (1–400).")
                return
        else:
            limit = 15

        self.status_var.set(f"Running {algo}…")
        self.stats_var.set("")
        self.running = True

        # Generator
        if   algo == "BFS":          gen = self.engine.bfs(self.start_pos, self.goal_pos, self.walls)
        elif algo == "DFS":          gen = self.engine.dfs(self.start_pos, self.goal_pos, self.walls)
        elif algo == "UCS":          gen = self.engine.ucs(self.start_pos, self.goal_pos, self.walls)
        elif algo == "DLS":          gen = self.engine.dls(self.start_pos, self.goal_pos, self.walls, limit)
        elif algo == "IDDFS":        gen = self.engine.iddfs(self.start_pos, self.goal_pos, self.walls)
        elif algo == "Bidirectional":gen = self.engine.bidirectional(self.start_pos, self.goal_pos, self.walls)

        final     = None
        explored_count = 0
        t0        = time.time()

        try:
            for node, explored, step, tag in gen:
                r, c = node.row, node.col

                if algo == "Bidirectional" and isinstance(explored, tuple):
                    ex_s, ex_g = explored
                    explored_count = len(ex_s) + len(ex_g)
                    if tag == "MEET":
                        self._set_cell(r, c, COLOR_MEETING)
                        final = node
                        break
                    elif tag == "S":
                        self._set_cell(r, c, COLOR_EXPLORED_S)
                    elif tag == "G":
                        self._set_cell(r, c, COLOR_EXPLORED_G)
                else:
                    explored_count = step
                    self._set_cell(r, c, COLOR_EXPLORED_S)

                    # DLS: show depth in status bar live
                    if tag:
                        self.status_var.set(f"{algo} | Step {step} | {tag}")

                    if node.get_pos() == self.goal_pos:
                        final = node
                        break

                self.root.update()
                time.sleep(self.speed_var.get())

        except Exception as ex:
            print(f"[Error] {ex}")
            import traceback; traceback.print_exc()

        elapsed = time.time() - t0

        if final:
            plen = self._draw_path(final)
            self.status_var.set(f"✅  Goal Found!   ({algo})")
            self.stats_var.set(
                f"Nodes explored : {explored_count}\n"
                f"Path length    : {plen} steps\n"
                f"Time taken     : {elapsed:.3f} s"
            )
        else:
            self.status_var.set(f"❌  No Path Found.   ({algo})")
            self.stats_var.set(
                f"Nodes explored : {explored_count}\n"
                f"Time taken     : {elapsed:.3f} s"
            )

        self.running = False

    # DRAW PATH

    def _draw_path(self, node):
        path = []

        if hasattr(node, "special_bidirectional_match"):
            # Goal half walk from meeting point's goal-side node up to goal
            cur = node.special_bidirectional_match
            while cur:
                path.append(cur.get_pos())
                cur = cur.parent

        # Start half
        cur = node
        while cur:
            path.append(cur.get_pos())
            cur = cur.parent

        # De-duplicate preserving order
        seen, uniq = set(), []
        for pos in path:
            if pos not in seen:
                seen.add(pos)
                uniq.append(pos)

        for r, c in uniq:
            self._set_cell(r, c, COLOR_PATH)
            self.root.update()
            time.sleep(0.03)

        return len(uniq)


# Entry Point
if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#eceff1")
    PathfinderGUI(root)
    root.mainloop()