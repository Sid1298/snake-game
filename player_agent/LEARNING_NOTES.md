# 🐍 Learning Notes: Teaching a Snake to Play Itself with Deep Q-Learning

> My notes on how the agent in `player_agent/` works: the RL ideas behind it, what each piece does, and where it lives in the code.
>
> **TL;DR:** The snake learns with **Deep Q-Learning (DQN)**, a **Reinforcement Learning** algorithm. A neural network learns to predict *"how good is each move right now?"*, and the snake picks the move with the highest predicted value. Nobody tells it the rules or a strategy. It works them out from points: **+10 for eating, −10 for dying**.

---

## 0. Map of the code

| File | What it is, in RL terms |
|---|---|
| [snake_env.py](snake_env.py) | The **environment**: the game rules, the rewards, and what the agent "sees" (the state) |
| [agent.py](agent.py) | The **agent**: the Q-network, replay memory, exploration and the learning update |
| [train.py](train.py) | The **training loop**: play, store experience, learn, repeat |
| [play.py](play.py) | **Evaluation**: play greedily with the learned network and no exploration |
| [render.py](render.py) | A window so a human can watch. It plays no part in learning |

---

## 1. The big picture: what is Reinforcement Learning?

There are three main flavours of machine learning:

- **Supervised learning**: "Here's an input, here's the right answer, learn the mapping." *(We don't have right answers for snake moves.)*
- **Unsupervised learning**: "Here's data, find structure in it."
- **Reinforcement Learning (RL)**: "Here's a world. Act in it, and you'll get points sometimes. Figure out how to get more points." ✅ *This is us.*

The RL loop, which every RL algorithm is built around:

```
        ┌──────────── action aₜ ────────────┐
        │                                   ▼
   ┌─────────┐                        ┌─────────────┐
   │  AGENT  │                        │ ENVIRONMENT │
   │ (brain) │                        │ (the game)  │
   └─────────┘                        └─────────────┘
        ▲                                  │
        └──── new state sₜ₊₁, reward rₜ ────┘
```

### Vocabulary ↔ our snake game

| RL term | Meaning | In our snake | Code |
|---|---|---|---|
| **Environment** | The world the agent acts in | The wrap-around snake board | `SnakeEnv`, [snake_env.py:29](snake_env.py#L29) |
| **State** `s` | What the agent observes | A vector of 20 numbers describing the surroundings | `observe()`, [snake_env.py:129](snake_env.py#L129) |
| **Action** `a` | A choice the agent makes | `0` = straight, `1` = turn right, `2` = turn left | [snake_env.py:21-23](snake_env.py#L21-L23) |
| **Reward** `r` | Immediate feedback | +10 food, −10 death, 0 otherwise | [snake_env.py:25-26](snake_env.py#L25-L26) |
| **Episode** | One full game, start to finish | Until the snake bites itself or starves | `reset()`, [snake_env.py:38](snake_env.py#L38) |
| **Step / transition** | One tick: `(s, a, r, s', done)` | One move of the snake | `step()`, [snake_env.py:58](snake_env.py#L58) |
| **Policy** `π(s)` | The agent's rule for choosing actions | "Pick the move with the highest Q-value" | `act()`, [agent.py:84](agent.py#L84) |

> 💡 **Key insight:** The agent never sees the code of the rules. It only experiences *"I was in situation X, did Y, got reward Z, and ended up in situation W."* Everything it learns comes from millions of these tuples.

---

## 2. Return and discounting: "how good is the future?"

The agent shouldn't just grab the biggest *immediate* reward. Most moves give 0, and only some lead to food later. So we care about the **return**, the sum of all future rewards:

$$G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \dots$$

**γ (gamma)** is the **discount factor**, between 0 and 1. In our code **γ = 0.95** ([agent.py:54](agent.py#L54)).

Why discount?
- A reward now is more certain than one far in the future.
- It keeps the sum finite.
- It makes the agent prefer *faster* routes to food.

📝 **Rule of thumb:** the "effective horizon" is about `1 / (1 − γ)` = `1 / 0.05` = **~20 steps**. The agent mostly plans about 20 moves ahead.

| Food is k steps away | Its value today = 10 × 0.95ᵏ |
|---|---|
| 1 | 9.5 |
| 5 | 7.7 |
| 10 | 6.0 |
| 20 | 3.6 |
| 50 | 0.8 |

So food that is 5 steps away "feels" better than food 20 steps away. That's what pushes the snake to take short paths.

---

## 3. Q-values: the core idea of Q-Learning

**Q(s, a)** = *"If I'm in state `s`, take action `a`, and then play well afterwards, what return do I expect?"*

If we knew the true Q for every state, playing perfectly would be trivial:

$$\pi(s) = \arg\max_a Q(s, a)$$

i.e. *pick the action with the highest Q-value*. That's literally what `act()` does at [agent.py:88-90](agent.py#L88-L90):

```python
q_values = self.policy(torch.from_numpy(state).unsqueeze(0))   # 3 numbers: Q(s, straight), Q(s, right), Q(s, left)
return int(q_values.argmax(dim=1).item())                        # pick the biggest
```

The whole learning problem becomes: **estimate Q accurately.**

---

## 4. The Bellman equation: how Q can be learned

Q-values satisfy a neat recursive relationship, the **Bellman equation**:

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

In words: *"The value of doing `a` now = the reward I get immediately + the discounted value of the best thing I can do from where I land."*

This is the magic trick. We don't need to know the whole future. We just need to make each Q-value **consistent with the next one**. If we keep nudging Q towards `r + γ·max Q(s')`, the correct values spread backwards from the rewards over time:

```
Episode 1:   Q learns "the step INTO food is good"
Episode N:   Q learns "the step before that is good too" (because its next state has high Q)
Episode 2N:  ...and the step before that...
```

### Classic (tabular) Q-learning

In the textbook version, Q is a big lookup table and each step does:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target}} - Q(s,a)\Big]$$

The bracket is called the **TD error** (Temporal Difference error). It measures how wrong the current guess was. `α` is the learning rate.

### ❌ Why a table doesn't work here

Our board is 22 × 15 = 330 cells. The number of possible board layouts (snake shapes × food positions) is astronomically large. We'd never visit most states even once, and a table can't generalise: it doesn't know that two *similar* situations should have similar values.

### ✅ Solution: replace the table with a neural network

The network takes a state and outputs a Q-value for every action. Similar states get similar outputs automatically, so the network **generalises**. Q-learning with a neural network is **Deep Q-Learning**, and the network is a **Deep Q-Network (DQN)**.

---

## 5. The Q-network

📍 [agent.py:10-22](agent.py#L10-L22)

```
   20 inputs (state)
        │
   Linear 20 → 256  + ReLU
        │
   Linear 256 → 256 + ReLU
        │
   Linear 256 → 3
        │
   [Q(straight), Q(right), Q(left)]
```

- It's a plain **MLP** (multi-layer perceptron) with about 72k parameters. That's small, so it trains fine on a CPU.
- There is **one output per action**, so a single forward pass scores all 3 moves at once. That's the standard DQN design.
- There is **no activation on the last layer**, because Q-values can be any real number, including negative ones.

---

## 6. The training loop, step by step

📍 [train.py:76-83](train.py#L76-L83). This is the heart of everything:

```python
state = env.reset()
while not done:
    action = agent.act(state)                                   # ① choose (ε-greedy)
    next_state, reward, done, info = env.step(action)           # ② act in the world
    agent.remember(state, action, reward, next_state, done)     # ③ store the experience
    agent.learn()                                               # ④ learn from a random batch
    state = next_state
```

| # | What happens | Where |
|---|---|---|
| ① | Pick an action, sometimes randomly (explore), otherwise the best Q (exploit) | [agent.py:84-90](agent.py#L84-L90) |
| ② | The game moves the snake and returns a reward and the new state | [snake_env.py:58-90](snake_env.py#L58-L90) |
| ③ | Save `(s, a, r, s', done)` into replay memory | [agent.py:91-93](agent.py#L91-L93) |
| ④ | Sample 256 old experiences and do one gradient step towards the Bellman target | [agent.py:95-117](agent.py#L95-L117) |

The sections below explain the four tricks that make ①–④ actually work.

---

## 7. Trick #1: ε-greedy exploration

**The dilemma (exploration vs. exploitation):** if the agent always picks what it *currently* thinks is best, it never tries anything new and may never discover that food exists. If it always acts randomly, it never uses what it learned.

**The fix:** with probability **ε** take a random action, otherwise take the best one.

📍 [agent.py:79-87](agent.py#L79-L87)

```python
@property
def epsilon(self):
    progress = min(self.total_steps / self.eps_decay_steps, 1.0)
    return self.eps_start + progress * (self.eps_end - self.eps_start)

def act(self, state, explore=True):
    if explore and random.random() < self.epsilon:
        return random.randrange(self.n_actions)     # explore
    ...                                             # exploit (argmax Q)
```

**Schedule:** ε starts at **1.0** (100% random) and falls linearly to **0.01** (1% random) over **50,000 steps** ([agent.py:60-62](agent.py#L60-L62)).

```
ε
1.0 ┤█
    │ ██
    │   ███
0.5 ┤      ████
    │          ████
    │              ████
0.01┤                  ██████████████████████
    └──────────────────┬─────────────────────── steps
                     50k
```

At **test time** ([play.py:42](play.py#L42)) we call `act(state, explore=False)`, which is pure exploitation with no randomness.

> 💡 **This explains the training curve I saw.** In the first run, the average score sat around 1–4 for 450 episodes and then jumped to ~60 between episodes 500 and 650. The network had been learning all along, but random moves (high ε) kept killing the snake. Once ε got small, the learned policy was finally allowed to show what it knew.

---

## 8. Trick #2: Experience replay

**Problem:** if you train on transitions in the order they happen, consecutive samples are almost identical (step 41 looks just like step 40). Neural networks train badly on highly correlated data. They overfit to "what just happened" and forget older lessons. Each experience would also only be used once, which is wasteful.

**Fix:** store every transition in a big memory buffer and train on **random mini-batches** drawn from it.

📍 `ReplayBuffer`, [agent.py:25-44](agent.py#L25-L44)

```python
self.memory = deque(maxlen=capacity)        # capacity = 100,000; the oldest are dropped automatically
...
batch = random.sample(self.memory, batch_size)   # 256 random past experiences
```

Benefits:
- 🔀 **Breaks correlation**: a batch mixes early-game, late-game, good and bad moments.
- ♻️ **Data efficiency**: each transition can be learned from many times.
- 🧠 **Less forgetting**: rare events (like dying in a weird spot) keep getting replayed.

There's also a **warm-up**. Learning doesn't start until there are 1,000 transitions in memory ([agent.py:97-98](agent.py#L97-L98)), so the first batches aren't tiny and repetitive.

> 📌 Because of replay, DQN is an **off-policy** algorithm: it learns from data collected by older versions of itself.

---

## 9. Trick #3: The target network

**Problem: chasing a moving target.** The training target is `r + γ · max Q(s')`. But Q comes from the *same network we're updating*. Every gradient step changes the target too. It's like a dog chasing its own tail, and training can oscillate or blow up.

**Fix:** keep a **second copy** of the network, the **target network**, frozen for a while. Use it to compute targets, and copy the main ("policy") network's weights into it every 1,000 steps.

📍 Created at [agent.py:72-74](agent.py#L72-L74), synced at [agent.py:115-116](agent.py#L115-L116):

```python
if self.total_steps % self.target_sync_steps == 0:      # every 1,000 steps
    self.target.load_state_dict(self.policy.state_dict())
```

| Network | Role | Updated |
|---|---|---|
| `self.policy` | Chooses actions and is trained by gradient descent | Every step |
| `self.target` | Provides stable Q(s') values for the targets | Copied over every 1,000 steps |

---

## 10. Trick #4: Double DQN

**Problem: overestimation bias.** Vanilla DQN uses `max_a' Q_target(s', a')`. Taking the **max over noisy estimates** is biased upwards: if any action's estimate is accidentally too high, the max picks exactly that one. Over many updates, Q-values drift to be overly optimistic.

**Fix (Double DQN):** split the job in two:
1. The **policy** network *chooses* the best next action.
2. The **target** network *evaluates* how good that action is.

It's unlikely both networks are wrong in the same direction at the same time, so the bias mostly cancels.

📍 [agent.py:103-107](agent.py#L103-L107)

```python
with torch.no_grad():
    next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)   # policy net PICKS
    next_q = self.target(next_states).gather(1, next_actions).squeeze(1)  # target net SCORES
    target = rewards + self.gamma * next_q * (1.0 - dones)
```

| | Picks a' with | Scores a' with |
|---|---|---|
| Vanilla DQN | target net | target net |
| **Double DQN (ours)** | **policy net** | **target net** |

---

## 11. The learning update, line by line

📍 `learn()`, [agent.py:95-117](agent.py#L95-L117)

```python
states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)   # (a) random batch

q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)                  # (b) Q for the actions actually taken

with torch.no_grad():                                                                # (c) build the targets (no gradients)
    next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
    next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
    target = rewards + self.gamma * next_q * (1.0 - dones)

loss = nn.functional.smooth_l1_loss(q, target)                                       # (d) how wrong were we?
self.optimizer.zero_grad()
loss.backward()                                                                      # (e) backprop
nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)                             # (f) safety clamp
self.optimizer.step()                                                                # (g) update weights (Adam, lr = 0.001)
```

Notes on each piece:

- **(b) `gather`**: the network outputs 3 Q-values per state, but we only have feedback for the action we *actually took*, so we pick out that one column.
- **(c) `(1.0 - dones)`**: ⚠️ important! If the episode ended (snake died), there is **no future**, so the target is just the reward. Multiplying by `1 - done` zeros out the future term for terminal states. Forget this and the agent thinks death leads somewhere.
- **(c) `torch.no_grad()`**: the target is treated as a fixed "label". We don't want gradients flowing through it.
- **(d) Huber loss (`smooth_l1_loss`)**: it behaves like squared error for small errors and like absolute error for big ones. One shocking transition (e.g. a −10 surprise) won't produce a huge gradient that wrecks the network.
- **(f) Gradient clipping**: a second safety net against exploding updates.

### 🧮 Worked example

Say γ = 0.95, and the target network estimates the best next move is worth 8.0.

| Situation | Target computation | Target |
|---|---|---|
| Normal move | 0 + 0.95 × 8.0 × (1 − 0) | **7.6** |
| Ate food | 10 + 0.95 × 8.0 × (1 − 0) | **17.6** |
| Died | −10 + 0.95 × 8.0 × (1 − **1**) | **−10.0** |

If the policy net had predicted Q = 5.0 for that normal move, the TD error is 7.6 − 5.0 = 2.6. The gradient step nudges the prediction up towards 7.6.

---

## 12. State design: what the snake "sees"

The neural network is only as good as its inputs. Here we use **hand-crafted features** instead of raw pixels. That makes learning much faster: minutes on a CPU instead of hours on a GPU.

📍 `observe()`, [snake_env.py:129-169](snake_env.py#L129-L169). It produces **20 numbers**:

| # | Feature | Values | Why it helps |
|---|---|---|---|
| 3 | **Danger**: is the next cell straight / right / left part of my body? | 0 or 1 | Immediate survival |
| 3 | **Free space**: fraction of the board reachable after each move (flood fill) | 0 – 1 | Avoids trapping itself in a pocket of its own body |
| 3 | **Body distance**: how far until I hit my body looking straight / right / left | 0 – 1 | Earlier warning than "danger" |
| 4 | **Food direction**: is food ahead / behind / to my right / to my left? | 0 or 1 | Where to go |
| 2 | **Food offset**: forward and rightward distance to food | about −1 to 1 | How far to go |
| 4 | **Heading**: one-hot of UP / RIGHT / DOWN / LEFT | 0 or 1 | The board is wider than tall, so direction matters a bit |
| 1 | **Length**: body length / board size | 0 – 1 | Bigger snake → be more careful |

### Design ideas worth remembering

**🧭 Relative (egocentric) frame.** Features are expressed as *straight / right / left*, from the snake's point of view, not as north/south/east/west. Also, actions are relative ([snake_env.py:21-22](snake_env.py#L21-L22)). This means:
- A lesson learned while heading up automatically applies while heading left.
- The snake can **never** choose the illegal "reverse into myself" move, because it simply isn't an action.

The food position is rotated into this frame using dot products with the heading vector ([snake_env.py:145-150](snake_env.py#L145-L150)).

**🌐 Wrap-around awareness.** On a torus-like board, food at the far right edge may actually be *2 steps to the left*. `_wrapped_delta` ([snake_env.py:97-100](snake_env.py#L97-L100)) always measures the **shortest** way round:

```python
d = (b - a) % size
return d - size if d > size / 2 else d
```

**🫧 Flood fill (free space).** `_reachable_cells` ([snake_env.py:102-117](snake_env.py#L102-L117)) counts how many cells the snake could still reach after a move. This is the main defence against the classic late-game death of curling into a dead end. Note that this is *information*, not a rule: the network still has to *learn* that low free space is bad.

> 🤔 **Is this "cheating"?** It's a design trade-off. Richer features mean faster learning but more human knowledge baked in. The "purest" version would feed the raw grid into a **CNN**. That needs far more training, but the agent discovers its own features. A good experiment for later!

---

## 13. Reward design

📍 [snake_env.py:25-26](snake_env.py#L25-L26) and `step()`, [snake_env.py:58-90](snake_env.py#L58-L90)

| Event | Reward | Ends episode? |
|---|---|---|
| Eat food | **+10** | No |
| Hit own body | **−10** | Yes (`self_hit`) |
| 660 steps with no food | **−10** | Yes (`starved`) |
| Any other move | **0** | No |

- **Sparse reward**: most steps give 0. The Bellman equation and discounting do the work of spreading credit back to the moves that *led to* food. This is the **credit assignment problem**.
- **Starvation limit** ([snake_env.py:34](snake_env.py#L34), = 2 × 330 cells). The real game has no timer, but without one an agent can learn "spin in a circle forever = never die = never get −10." This is **reward hacking**: the agent finds a loophole that maximises reward without doing what we wanted.
- We deliberately **don't** reward "moving closer to food" (**reward shaping**). Shaping can speed up learning, but it can also be exploited (wiggling back and forth near food). The food-direction features were enough here.

---

## 14. Reading the training output

```
ep   400 | score   2 | avg100   2.74 | best   9 | eps 0.371 | ...
ep   500 | score   8 | avg100   6.92 | best  22 | eps 0.176 | ...
ep   550 | score  31 | avg100  18.82 | best  93 | eps 0.010 | ...
ep   650 | score  35 | avg100  61.56 | best 117 | eps 0.010 | ...
```

| Column | Meaning |
|---|---|
| `score` | Food eaten in that single episode (noisy!) |
| `avg100` | Average of the last 100 episodes. **This is the number to watch** |
| `best` | Best single game so far |
| `eps` | Current exploration rate ε |
| `steps` | Total environment steps so far, which drives the ε schedule |

**Checkpoints** ([train.py:94-97](train.py#L94-L97)):
- `dqn_latest.pt` is saved regularly, for `--resume`.
- `dqn_best.pt` is saved when `avg100` hits a new high, **but only once ε < 0.1**. Before that, scores mostly reflect randomness, not skill.

Training scores are also always a bit *lower* than test scores, because training still has 1% random moves. In my run: training avg ≈ 63, greedy test avg ≈ 91.

---

## 15. Hyperparameter cheat sheet

📍 `DQNAgent.__init__`, [agent.py:50-63](agent.py#L50-L63)

| Hyperparameter | Value | What it controls | If you increase it… |
|---|---|---|---|
| `gamma` (γ) | 0.95 | How far ahead the agent cares | More long-term planning, but slower and less stable learning |
| `lr` | 1e-3 | Adam step size | Faster but jumpier learning; too high → diverges |
| `batch_size` | 256 | Experiences per update | Smoother gradients, more compute per step |
| `buffer_size` | 100,000 | Replay memory size | Remembers older experience; too big → learns from stale behaviour |
| `warmup_steps` | 1,000 | Steps collected before learning starts | Less noisy start |
| `target_sync_steps` | 1,000 | How often the target net is refreshed | More stable, but slower to absorb new knowledge |
| `eps_start → eps_end` | 1.0 → 0.01 | Exploration range | — |
| `eps_decay_steps` | 50,000 | How long exploration lasts | More exploration and a later "takeoff" |
| `hidden` | 256 | Network width | More capacity, slower |

---

## 16. Concept summary (one-page cheatsheet)

```
GOAL      maximise   G = r₀ + γr₁ + γ²r₂ + ...

Q(s,a)    = expected G after taking a in s, then playing well
POLICY    a = argmax_a Q(s,a)                                (ε-greedy while training)

BELLMAN   Q(s,a) = r + γ · max_a' Q(s',a')

DQN       Q is a neural network; train it so that
          Q_policy(s,a)  ≈  r + γ · Q_target(s', argmax_a' Q_policy(s',a')) · (1 − done)
                                     ╰── Double DQN: policy picks, target scores ──╯

STABILITY TRICKS
  ① ε-greedy        explore early, exploit later
  ② replay buffer   train on random past experience (decorrelated, reusable)
  ③ target network  frozen copy for stable targets, synced every N steps
  ④ Double DQN      reduces overestimation of Q
  ⑤ Huber loss + gradient clipping   no exploding updates
```

---

## 17. Pitfalls I should remember

- ⚠️ **Forgetting `(1 - done)`** → the agent thinks death has a future value.
- ⚠️ **No time limit** → the agent may learn to loop forever (reward hacking).
- ⚠️ **ε decays too fast** → the agent locks in a bad strategy before it has seen enough. **Too slow** → wasted time playing randomly.
- ⚠️ **Judging progress by single-episode scores** → far too noisy. Use the rolling average.
- ⚠️ **Train/test mismatch** → the environment must match the real game's rules (e.g. in `game.py`, running into your own *tail* is fatal, so it is here too: [snake_env.py:68-72](snake_env.py#L68-L72)).
- ⚠️ **Evaluating with exploration on** → always test with `explore=False`.

---

## 18. Ideas to explore next

1. **CNN on the raw grid**: remove the hand-crafted features and let the network learn what to look at.
2. **Dueling DQN**: split the network into "how good is this state" + "how much better is each action".
3. **Prioritised experience replay**: replay surprising transitions (big TD error) more often.
4. **n-step returns**: use several real rewards before bootstrapping, so credit spreads faster.
5. **Policy-gradient methods (PPO, A2C)**: learn the policy directly instead of Q-values. A whole other family of RL.
6. **Plot `models/dqn_log.csv`**: chart score vs. episode and watch the "takeoff" moment.

---

## 19. Glossary

| Term | Short definition |
|---|---|
| **Agent** | The learner and decision-maker |
| **Environment** | Everything the agent interacts with |
| **Episode** | One game from reset to game-over |
| **Return (G)** | Discounted sum of future rewards |
| **Discount factor (γ)** | How much future rewards are worth compared with immediate ones |
| **Q-value** | Expected return of an action in a state |
| **Policy (π)** | The mapping from states to actions |
| **Bellman equation** | A Q-value equals the immediate reward plus the discounted best next Q-value |
| **TD error** | Target − prediction; the "surprise" the agent learns from |
| **Bootstrapping** | Using your own estimate (Q(s')) to build the training target |
| **Off-policy** | Learning from data generated by a different (older) policy |
| **ε-greedy** | Random action with probability ε, otherwise the best action |
| **Experience replay** | Storing transitions and training on random samples of them |
| **Target network** | A slowly updated copy of the Q-network, used for stable targets |
| **Double DQN** | Using one network to choose and another to evaluate the next action |
| **Reward hacking** | The agent exploiting a loophole in the reward instead of solving the task |
| **Credit assignment** | Working out which past actions caused a later reward |
