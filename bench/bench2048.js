#!/usr/bin/env node
/*
  Offline 2048 benchmarking harness (Node.js, no DOM).
  Purpose: generate reproducible metrics (win rate, avg score, max-tile distribution)
  for multiple agents, including "LLM-improvement" proxies that do NOT call an LLM.

  Usage examples:
    node bench/bench2048.js --agent expectimax --games 200 --seed 123
    node bench/bench2048.js --agent llm_guarded --games 500 --seed 1
    node bench/bench2048.js --all --games 200 --seed 123

  Notes:
  - This harness mirrors core mechanics from index.html (slide/merge + random spawn 2/4).
  - "llm_*" agents here are algorithmic stand-ins for prompt+constraint pipelines.
*/

'use strict';

const SIZE = 4;
const ACTIONS = ['up', 'down', 'left', 'right'];

// ---------------- RNG (reproducible) ----------------
class XorShift32 {
  constructor(seed) {
    let s = Number(seed) >>> 0;
    if (s === 0) s = 0x9e3779b9;
    this.state = s;
  }
  nextU32() {
    // xorshift32
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
  float() {
    // [0,1)
    return this.nextU32() / 0x100000000;
  }
  int(n) {
    return Math.floor(this.float() * n);
  }
  pick(arr) {
    return arr[this.int(arr.length)];
  }
}

// ---------------- Board utils ----------------
function cloneBoard(board) {
  return board.map(row => row.slice());
}

function boardFlat(board) {
  return board.flat();
}

function maxTile(board) {
  return Math.max(...boardFlat(board));
}

function countEmpty(board) {
  let e = 0;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      if (board[r][c] === 0) e++;
    }
  }
  return e;
}

function emptyCells(board) {
  const cells = [];
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      if (board[r][c] === 0) cells.push([r, c]);
    }
  }
  return cells;
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

function boardEquals(a, b) {
  for (let r = 0; r < SIZE; r++) {
    if (!arraysEqual(a[r], b[r])) return false;
  }
  return true;
}

// --------------- 2048 mechanics ---------------
function processLine(line) {
  // mirrors index.html: compact then merge once per adjacent equal pair
  const compact = line.filter(n => n !== 0);
  let merged = false;
  let gain = 0;
  for (let i = 0; i < compact.length - 1; i++) {
    if (compact[i] === compact[i + 1]) {
      compact[i] *= 2;
      gain += compact[i];
      compact.splice(i + 1, 1);
      merged = true;
    }
  }
  while (compact.length < SIZE) compact.push(0);
  return { line: compact, merged, gain };
}

function simulateMove(board, direction) {
  const next = cloneBoard(board);
  let moved = false;
  let merged = false;
  let gain = 0;

  if (direction === 'left' || direction === 'right') {
    for (let r = 0; r < SIZE; r++) {
      const row = next[r].slice();
      const line = direction === 'left' ? row : row.slice().reverse();
      const p = processLine(line);
      const newRow = direction === 'left' ? p.line : p.line.slice().reverse();
      merged = merged || p.merged;
      gain += p.gain;
      if (!arraysEqual(next[r], newRow)) {
        next[r] = newRow;
        moved = true;
      }
    }
  } else {
    for (let c = 0; c < SIZE; c++) {
      const col = [];
      for (let r = 0; r < SIZE; r++) col.push(next[r][c]);
      const line = direction === 'up' ? col : col.slice().reverse();
      const p = processLine(line);
      const newCol = direction === 'up' ? p.line : p.line.slice().reverse();
      merged = merged || p.merged;
      gain += p.gain;
      if (!arraysEqual(col, newCol)) {
        for (let r = 0; r < SIZE; r++) next[r][c] = newCol[r];
        moved = true;
      }
    }
  }

  return { moved, merged, gain, nextBoard: next };
}

function spawnRandomTile(board, rng) {
  const empties = emptyCells(board);
  if (empties.length === 0) return false;
  const [r, c] = rng.pick(empties);
  board[r][c] = rng.float() < 0.9 ? 2 : 4;
  return true;
}

function isGameOver(board) {
  // mirrors index.html: no empty and no adjacent equals
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const v = board[r][c];
      if (v === 0) return false;
      if (c < SIZE - 1 && v === board[r][c + 1]) return false;
      if (r < SIZE - 1 && v === board[r + 1][c]) return false;
    }
  }
  return true;
}

// ---------------- Evaluation ----------------
// weight matrix consistent with index.html (snake-ish, top-right is highest)
const WEIGHT_MATRIX = [
  [Math.pow(4, 15), Math.pow(4, 14), Math.pow(4, 13), Math.pow(4, 12)],
  [Math.pow(4, 8),  Math.pow(4, 9),  Math.pow(4, 10), Math.pow(4, 11)],
  [Math.pow(4, 7),  Math.pow(4, 6),  Math.pow(4, 5),  Math.pow(4, 4)],
  [Math.pow(4, 0),  Math.pow(4, 1),  Math.pow(4, 2),  Math.pow(4, 3)]
];

function calculateMonotonicity(board) {
  let mono = 0;
  for (let r = 0; r < SIZE; r++) {
    let inc = 0, dec = 0;
    for (let c = 0; c < SIZE - 1; c++) {
      if (board[r][c] <= board[r][c + 1]) inc++;
      if (board[r][c] >= board[r][c + 1]) dec++;
    }
    mono += Math.max(inc, dec);
  }
  for (let c = 0; c < SIZE; c++) {
    let inc = 0, dec = 0;
    for (let r = 0; r < SIZE - 1; r++) {
      if (board[r][c] <= board[r + 1][c]) inc++;
      if (board[r][c] >= board[r + 1][c]) dec++;
    }
    mono += Math.max(inc, dec);
  }
  return mono;
}

function calculateSmoothness(board) {
  let smooth = 0;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      const v = board[r][c];
      if (v === 0) continue;
      const lv = Math.log2(v);
      if (c < SIZE - 1 && board[r][c + 1] !== 0) smooth -= Math.abs(lv - Math.log2(board[r][c + 1]));
      if (r < SIZE - 1 && board[r + 1][c] !== 0) smooth -= Math.abs(lv - Math.log2(board[r + 1][c]));
    }
  }
  return smooth;
}

function evaluateBoard(board) {
  let score = 0;
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) {
      score += board[r][c] * WEIGHT_MATRIX[r][c];
    }
  }
  const empties = countEmpty(board);
  score += empties * 50000;
  score += calculateMonotonicity(board) * 10000;
  score += calculateSmoothness(board) * 1000;
  const mx = maxTile(board);
  if (board[0][SIZE - 1] === mx) score += mx * 10000;
  return score;
}

// ---------------- Agents ----------------
function validMoves(board) {
  const moves = [];
  for (const a of ACTIONS) {
    if (simulateMove(board, a).moved) moves.push(a);
  }
  return moves;
}

function anchorAtTopRight(board) {
  const mx = maxTile(board);
  return mx > 0 && board[0][SIZE - 1] === mx;
}

function rightColumnFull(board) {
  for (let r = 0; r < SIZE; r++) {
    if (board[r][SIZE - 1] === 0) return false;
  }
  return true;
}

function anchorGuard(board, proposedMove) {
  if (!anchorAtTopRight(board)) return proposedMove;
  if (proposedMove !== 'down') return proposedMove;
  if (rightColumnFull(board)) return proposedMove;

  // choose a safe alternative if possible
  for (const alt of ['up', 'right', 'left']) {
    if (simulateMove(board, alt).moved) return alt;
  }
  return proposedMove;
}

function agent_random(board, rng) {
  const moves = validMoves(board);
  if (moves.length === 0) return null;
  return rng.pick(moves);
}

// A proxy for a naive prompt: "prefer up/right, avoid down"
function agent_llm_naive(board) {
  const pref = ['right', 'up', 'left', 'down'];
  for (const a of pref) {
    if (simulateMove(board, a).moved) return a;
  }
  return null;
}

function agent_llm_guarded(board) {
  const move = agent_llm_naive(board);
  if (!move) return null;
  return anchorGuard(board, move);
}

// "LLM as reranker": generate candidates (like LLM suggestions) then pick best by heuristic
function agent_llm_rerank(board) {
  const candidates = validMoves(board);
  if (candidates.length === 0) return null;

  // Bias the candidate ordering to reflect "prompt preference"
  const order = ['right', 'up', 'left', 'down'];
  candidates.sort((a, b) => order.indexOf(a) - order.indexOf(b));

  let best = candidates[0];
  let bestScore = -Infinity;
  for (const a of candidates) {
    const sim = simulateMove(board, a);
    const guarded = anchorGuard(board, a);
    const sim2 = guarded === a ? sim : simulateMove(board, guarded);
    if (!sim2.moved) continue;
    const s = evaluateBoard(sim2.nextBoard) + sim2.gain;
    if (s > bestScore) {
      bestScore = s;
      best = guarded;
    }
  }
  return best;
}

// Hybrid: 1-ply expectimax-like selection (fast) used as a strong baseline
function agent_greedy_eval(board) {
  let best = null;
  let bestScore = -Infinity;
  for (const a of ACTIONS) {
    const sim = simulateMove(board, a);
    if (!sim.moved) continue;
    const s = evaluateBoard(sim.nextBoard) + sim.gain;
    if (s > bestScore) {
      bestScore = s;
      best = a;
    }
  }
  return best;
}

function expectimaxValue(board, depth, isPlayerTurn, rng, chanceSample = 4) {
  if (depth === 0) return evaluateBoard(board);

  if (isPlayerTurn) {
    let maxV = -Infinity;
    let any = false;
    for (const a of ACTIONS) {
      const sim = simulateMove(board, a);
      if (!sim.moved) continue;
      any = true;
      const v = expectimaxValue(sim.nextBoard, depth - 1, false, rng, chanceSample);
      if (v > maxV) maxV = v;
    }
    return any ? maxV : evaluateBoard(board);
  }

  const empties = emptyCells(board);
  if (empties.length === 0) return expectimaxValue(board, depth - 1, true, rng, chanceSample);

  // sample empties for speed, but deterministically with rng
  const sampled = [];
  const seen = new Set();
  const k = Math.min(chanceSample, empties.length);
  while (sampled.length < k) {
    const idx = rng.int(empties.length);
    if (seen.has(idx)) continue;
    seen.add(idx);
    sampled.push(empties[idx]);
  }

  let expected = 0;
  for (const [r, c] of sampled) {
    const b2 = cloneBoard(board);
    b2[r][c] = 2;
    expected += 0.9 * expectimaxValue(b2, depth - 1, true, rng, chanceSample);

    const b4 = cloneBoard(board);
    b4[r][c] = 4;
    expected += 0.1 * expectimaxValue(b4, depth - 1, true, rng, chanceSample);
  }
  return expected / sampled.length;
}

function agent_expectimax(board, rng, depth = 4) {
  let best = null;
  let bestV = -Infinity;
  for (const a of ACTIONS) {
    const sim = simulateMove(board, a);
    if (!sim.moved) continue;
    const v = expectimaxValue(sim.nextBoard, depth - 1, false, rng);
    if (v > bestV) {
      bestV = v;
      best = a;
    }
  }
  return best ?? agent_greedy_eval(board);
}

function runGreedyRollout(startBoard, rng, maxMoves = 150, epsilon = 0.2) {
  let board = cloneBoard(startBoard);
  let totalGain = 0;

  for (let t = 0; t < maxMoves; t++) {
    const moves = [];
    for (const a of ACTIONS) {
      const sim = simulateMove(board, a);
      if (!sim.moved) continue;
      moves.push({ a, score: evaluateBoard(sim.nextBoard) + sim.gain, next: sim.nextBoard, gain: sim.gain });
    }
    if (moves.length === 0) break;

    let chosen;
    if (rng.float() < epsilon) {
      chosen = rng.pick(moves);
    } else {
      moves.sort((x, y) => y.score - x.score);
      chosen = moves[0];
    }

    board = chosen.next;
    totalGain += chosen.gain;
    if (!spawnRandomTile(board, rng)) break;
    if (isGameOver(board)) break;
  }

  return totalGain + evaluateBoard(board) / 10000;
}

function agent_montecarlo(board, rng, simulations = 100) {
  let best = null;
  let bestV = -Infinity;

  for (const a of ACTIONS) {
    const sim = simulateMove(board, a);
    if (!sim.moved) continue;
    let total = 0;
    for (let k = 0; k < simulations; k++) {
      // use a derived RNG stream for each rollout for reproducibility
      const sub = new XorShift32(rng.nextU32());
      total += runGreedyRollout(sim.nextBoard, sub);
    }
    const v = total / simulations + evaluateBoard(sim.nextBoard) * 0.1;
    if (v > bestV) {
      bestV = v;
      best = a;
    }
  }

  return best ?? agent_greedy_eval(board);
}

const AGENTS = {
  random: (board, rng) => agent_random(board, rng),
  greedy_eval: (board, rng) => agent_greedy_eval(board),
  expectimax: (board, rng, cfg) => agent_expectimax(board, rng, cfg?.depth ?? 4),
  montecarlo: (board, rng, cfg) => agent_montecarlo(board, rng, cfg?.simulations ?? 100),
  llm_naive: (board, rng) => agent_llm_naive(board),
  llm_guarded: (board, rng) => agent_llm_guarded(board),
  llm_rerank: (board, rng) => agent_llm_rerank(board)
};

// --------------- Benchmark runner ---------------
function playOneGame(agentName, rng, cfg) {
  const agent = AGENTS[agentName];
  if (!agent) throw new Error(`Unknown agent: ${agentName}`);

  let board = Array.from({ length: SIZE }, () => Array(SIZE).fill(0));
  let score = 0;
  spawnRandomTile(board, rng);
  spawnRandomTile(board, rng);

  let steps = 0;
  while (!isGameOver(board) && steps < (cfg.maxSteps ?? 10000)) {
    const a = agent(board, rng, cfg);
    if (!a) break;
    const sim = simulateMove(board, a);
    if (!sim.moved) break;
    board = sim.nextBoard;
    score += sim.gain;
    if (!spawnRandomTile(board, rng)) break;
    steps++;
  }

  const mx = maxTile(board);
  return {
    score,
    steps,
    maxTile: mx,
    win2048: mx >= 2048
  };
}

function summarize(results) {
  const n = results.length;
  const wins = results.reduce((acc, r) => acc + (r.win2048 ? 1 : 0), 0);
  const avgScore = results.reduce((acc, r) => acc + r.score, 0) / n;
  const avgSteps = results.reduce((acc, r) => acc + r.steps, 0) / n;

  const dist = new Map();
  for (const r of results) {
    dist.set(r.maxTile, (dist.get(r.maxTile) ?? 0) + 1);
  }

  // Sort maxTile distribution
  const maxTileDist = Array.from(dist.entries())
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([tile, count]) => ({ tile: Number(tile), count, pct: count / n }));

  return {
    games: n,
    winRate2048: wins / n,
    avgScore,
    avgSteps,
    maxTileDist
  };
}

function parseArgs(argv) {
  const args = { games: 200, seed: 123, agent: 'expectimax', all: false, json: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--all') args.all = true;
    else if (a === '--json') args.json = true;
    else if (a === '--agent') args.agent = argv[++i];
    else if (a === '--games') args.games = Number(argv[++i]);
    else if (a === '--seed') args.seed = Number(argv[++i]);
    else if (a === '--depth') args.depth = Number(argv[++i]);
    else if (a === '--simulations') args.simulations = Number(argv[++i]);
    else if (a === '--maxSteps') args.maxSteps = Number(argv[++i]);
    else if (a === '--help' || a === '-h') args.help = true;
    else throw new Error(`Unknown arg: ${a}`);
  }
  return args;
}

function printHuman(agentName, summaryObj, elapsedMs) {
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  console.log(`Agent: ${agentName}`);
  console.log(`Games: ${summaryObj.games}`);
  console.log(`Win@2048: ${pct(summaryObj.winRate2048)}`);
  console.log(`Avg score: ${summaryObj.avgScore.toFixed(1)}`);
  console.log(`Avg steps: ${summaryObj.avgSteps.toFixed(1)}`);
  console.log(`Elapsed: ${elapsedMs.toFixed(0)} ms`);

  const top = summaryObj.maxTileDist
    .slice()
    .sort((a, b) => b.tile - a.tile)
    .slice(0, 6);
  console.log('Top max-tile frequencies (largest tiles):');
  for (const row of top) {
    console.log(`  ${row.tile}: ${row.count} (${pct(row.pct)})`);
  }
}

function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log('Usage: node bench/bench2048.js [--agent NAME | --all] [--games N] [--seed S] [--depth D] [--simulations K] [--json]');
    console.log('Agents:', Object.keys(AGENTS).join(', '));
    process.exit(0);
  }

  const agentList = args.all ? Object.keys(AGENTS) : [args.agent];

  const outputs = {};
  const start = Date.now();

  for (const agentName of agentList) {
    const rng = new XorShift32(args.seed);
    const cfg = {
      depth: args.depth,
      simulations: args.simulations,
      maxSteps: args.maxSteps
    };
    const results = [];
    for (let i = 0; i < args.games; i++) {
      // Derive a per-game RNG stream for stable replication
      const gameRng = new XorShift32(rng.nextU32());
      results.push(playOneGame(agentName, gameRng, cfg));
    }
    outputs[agentName] = summarize(results);
  }

  const elapsedMs = Date.now() - start;

  if (args.json) {
    console.log(JSON.stringify({ args, elapsedMs, outputs }, null, 2));
    return;
  }

  for (const agentName of agentList) {
    printHuman(agentName, outputs[agentName], elapsedMs);
    if (agentList.length > 1) console.log('');
  }
}

main();
