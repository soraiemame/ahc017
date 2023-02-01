use proconio::{input,marker::Usize1};
use std::collections::BinaryHeap;
use std::cmp::Reverse;

const T9: u32 = 1_000_000_000;

fn main() {
    get_time();
    let input = Input::from_input();
    solve(input);
}

fn solve(input: Input) {
    let mut rng = Xorshift::new();
    // let mut cs = State::new(&input);
    // let mut min_score = cs.cur_score;
    let dist = dijkstra(input.n, &input.es);
    let mut loop_cnt = 0;
    let mut res = vec![0;input.m];
    let mut day_count = vec![0;input.d];
    let mut now = 0;
    for i in 0..input.m {
        res[i] = now;
        day_count[now] += 1;
        if day_count[now] == input.k {
            now += 1;
        }
    }
    let mut min_score = score(&res,&dist,&input);
    eprintln!("before: {}",min_score);
    while get_time() < 5.9 {
        loop_cnt += 1;
        let ci = rng.rand(input.n);
        let pdi = res[ci];
        day_count[pdi] -= 1;
        let mut di = rng.rand(input.d);
        while day_count[di] == input.k {
            di = rng.rand(input.d);
        }
        day_count[di] += 1;
        res[ci] = di;
        let cur_score = score(&res,&dist,&input);
        // let cur_score = cs.try_apply(ci, di);
        if !chmin!(min_score,cur_score) {
            res[ci] = pdi;
            day_count[pdi] += 1;
            day_count[di] -= 1;
            // cs.apply(ci,di);
        }
    }
    eprintln!("score: {}",min_score);
    eprintln!("loops: {}",loop_cnt);
    print_ans(res);
}

fn score(ans: &Vec<usize>,dist: &Vec<Vec<u32>>,input: &Input) -> u64 {
    let mut turn_es = vec![vec![];input.d];
    for i in 0..input.m {
        for j in 0..input.d {
            if j != ans[i] {
                turn_es[j].push(input.es[i]);
            }
        }
    }
    let mut res = 0.0;
    for i in 0..input.d {
        let cur_dist = dijkstra(input.n, &turn_es[i]);
        let mut cnt = 0u64;
        for j in 0..input.n {
            for k in j + 1..input.n {
                cnt += (cur_dist[j][k] - dist[j][k]) as u64 * 2;
            }
        }
        let fk = cnt as f64 / (input.n as f64 * (input.n - 1) as f64);
        res += fk;
    }
    (res / input.d as f64 * 1e3).round() as u64
}

struct State {
    day_count: Vec<usize>,
    dist: Vec<Vec<u32>>,
    cur_score: u64,
    ops: Vec<usize>,
    turn_es: Vec<Vec<(usize,usize,u32)>>,
    cur_dists: Vec<Vec<Vec<u32>>>,
    fks: Vec<f64>,
}

impl State {
    fn new(input: &Input) -> Self {
        let dist = dijkstra(input.n,&input.es);
        let mut ops = vec![0; input.m];
        let mut day_count = vec![0; input.d];
        let mut now = 0;
        for i in 0..input.m {
            ops[i] = now;
            day_count[now] += 1;
            if day_count[now] == input.k {
                now += 1;
            }
        }
        let mut turn_es = vec![vec![];input.d];
        for i in 0..input.m {
            for j in 0..input.d {
                if j != ops[i] {
                    turn_es[j].push(input.es[i]);
                }
            }
        }
        let mut res = 0.0;
        let mut cur_dists = vec![];
        let mut fks = vec![];
        cur_dists.reserve(input.d);
        fks.reserve(input.d);
        for i in 0..input.d {
            let cur_dist = dijkstra(input.n, &turn_es[i]);
            let mut cnt = 0u64;
            for j in 0..input.n {
                for k in j + 1..input.n {
                    cnt += (cur_dist[j][k] - dist[j][k]) as u64 * 2;
                }
            }
            let fk = cnt as f64 / (input.n as f64 * (input.n - 1) as f64);
            res += fk;
            cur_dists.push(cur_dist);
            fks.push(fk);
        }
        let cur_score = (res / input.d as f64 * 1e3).round() as u64;
        Self {
            day_count,
            dist,
            cur_score,
            ops,
            turn_es,
            cur_dists,
            fks
        }
    }
    fn try_apply(&self,input: &Input,ci: usize,di: usize) -> u64 {
        // let mut res = 0.0;
        // res -= self.fks[self.ops[ci]];
        // res -= self.fks[self.ops[ci]];
        todo!()
    }
    fn apply(&mut self,input: &Input,ci: usize,di: usize) {
        let pdi = self.ops[ci];
        self.day_count[pdi] -= 1;
        // day_count[di] += 1;
        // res[ci] = di;
    }
}

fn dijkstra(n: usize,es: &Vec<(usize,usize,u32)>) -> Vec<Vec<u32>> {
    let mut g = vec![vec![];n];
    for e in es {
        g[e.0].push((e.1,e.2));
        g[e.1].push((e.0,e.2));
    }
    let mut res = vec![vec![T9;n];n];
    let mut que = BinaryHeap::new();
    for i in 0..n {
        res[i][i] = 0;
        que.push((Reverse(0),i));
        while let Some((Reverse(d),p)) = que.pop() {
            if res[i][p] < d {
                continue;
            }
            for e in &g[p] {
                if chmin!(res[i][e.0],d + e.1) {
                    que.push((Reverse(res[i][e.0]),e.0));
                }
            }
        }
    }
    res
}

fn print_ans(ans: Vec<usize>) {
    for i in 0..ans.len() {
        print!("{}", ans[i] + 1);
        print!("{}", if i == ans.len() { "\n" } else { " " });
    }
}

struct Input {
    n: usize,
    m: usize,
    d: usize,
    k: usize,
    es: Vec<(usize, usize, u32)>,
    g: Vec<Vec<(usize, usize, u32)>>,
    p: Vec<(u32, u32)>,
}

impl Input {
    fn from_input() -> Self {
        input! {
            n: usize,m: usize,d: usize,k: usize,
            es: [(Usize1,Usize1,u32);m]
        };
        let mut g = vec![vec![]; n];
        for i in 0..m {
            g[es[i].0].push((i, es[i].1, es[i].2));
            g[es[i].1].push((i, es[i].0, es[i].2));
        }
        input! {
            p: [(u32,u32);n]
        };
        Self {
            n,
            m,
            d,
            k,
            es,
            g,
            p,
        }
    }
}


pub fn get_time() -> f64 {
	static mut STIME: f64 = -1.0;
	let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
	let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
	unsafe {
		if STIME < 0.0 {
			STIME = ms;
		}
		// ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
		#[cfg(feature="local")]
		{
			(ms - STIME) * 1.5
		}
		#[cfg(not(feature="local"))]
		{
			(ms - STIME)
		}
	}
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Xorshift {
    seed: u64,
}
impl Xorshift {
    #[allow(dead_code)]
    pub fn new() -> Xorshift {
        Xorshift {
            seed: 0xf0fb588ca2196dac,
        }
    }
    #[allow(dead_code)]
    pub fn with_seed(seed: u64) -> Xorshift {
        Xorshift { seed: seed }
    }
    #[inline]
    #[allow(dead_code)]
    pub fn next(&mut self) -> u64 {
        self.seed = self.seed ^ (self.seed << 13);
        self.seed = self.seed ^ (self.seed >> 7);
        self.seed = self.seed ^ (self.seed << 17);
        self.seed
    }
    #[inline]
    #[allow(dead_code)]
    pub fn rand(&mut self, m: usize) -> usize {
        self.next() as usize % m
    }
    #[inline]
    #[allow(dead_code)]
    // 0.0 ~ 1.0
    pub fn randf(&mut self) -> f64 {
        use std::mem;
        const UPPER_MASK: u64 = 0x3FF0000000000000;
        const LOWER_MASK: u64 = 0xFFFFFFFFFFFFF;
        let tmp = UPPER_MASK | (self.next() & LOWER_MASK);
        let result: f64 = unsafe { mem::transmute(tmp) };
        result - 1.0
    }
}

#[macro_use]
mod macros {
    #[allow(unused_macros)]
    #[cfg(debug_assertions)]
    #[macro_export]
    macro_rules! debug {
        ( $x: expr, $($rest:expr),* ) => {
            eprint!(concat!(stringify!($x),": {:?}, "),&($x));
            debug!($($rest),*);
        };
        ( $x: expr ) => { eprintln!(concat!(stringify!($x),": {:?}"),&($x)); };
        () => { eprintln!(); };
    }
    #[allow(unused_macros)]
    #[cfg(not(debug_assertions))]
    #[macro_export]
    macro_rules! debug {
        ( $($x: expr),* ) => {};
        () => {};
    }
    #[macro_export]
    macro_rules! chmin {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_min = min!($($cmps),+);
            if $base > cmp_min {
                $base = cmp_min;
                true
            } else {
                false
            }
        }};
    }

    #[macro_export]
    macro_rules! chmax {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_max = max!($($cmps),+);
            if $base < cmp_max {
                $base = cmp_max;
                true
            } else {
                false
            }
        }};
    }

    #[macro_export]
    macro_rules! min {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::min($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::min($a, min!($($rest),+))
        }};
    }
    #[macro_export]
    macro_rules! max {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::max($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::max($a, max!($($rest),+))
        }};
    }

    #[macro_export]
    macro_rules! mat {
        ($e:expr; $d:expr) => { vec![$e; $d] };
        ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
    }
}
