use proconio::{input, marker::Usize1};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

const T9: u32 = 1_000_000_000;

fn main() {
    get_time();
    let input = Input::from_input();
    // solve(input);
    // solve2(input);
    solve3(input);
}

// a(500,500),b(500,1000),p
fn deg(x: u32, y: u32) -> f64 {
    if x == 500 && y == 500 {
        return 0.0;
    }
    let theta = (y as f64 - 500.0).atan2(x as f64 - 500.0) + PI;
    theta
}
fn solve2(input: Input) {
    let mut res = vec![0; input.m];
    let mut day_count = vec![0; input.d];
    let mut ps = vec![0; input.n];
    let mut turn_ps = vec![vec![]; input.d];
    for i in 0..input.n {
        let p = input.p[i];
        let theta = deg(p.0, p.1);
        // eprintln!("{} {} {}",p.0,p.1,theta);
        let c = (theta / (2.0 * PI / input.d as f64 / 2.0)).floor() as usize;
        ps[i] = c % input.d;
        turn_ps[c % input.d].push(i);
    }
    let mut turn_es = vec![vec![]; input.d];
    for i in 0..input.m {
        let idx = ps[input.es[i].0];
        if day_count[idx] != input.k {
            res[i] = idx;
            turn_es[idx].push(i);
            day_count[idx] += 1;
        }
    }
    let mut is_unused = vec![false;input.m];
    let mut unused = vec![vec![];input.d];
    for i in 0..input.d {
        let mut uf = acl::Dsu::new(input.n);
        for &j in &turn_es[i] {
            if !uf.same(input.es[j].0,input.es[j].1) {
                uf.merge(input.es[j].0, input.es[j].1);
                unused[i].push(j);
                is_unused[j] = true;
            }
        }
    }
    for i in 0..input.d {
        let mut rem = vec![];
        for &j in &turn_es[i] {
            if !is_unused[j] {
                rem.push(j);
            }
        }
        turn_es[i] = rem;
    }
    for i in 0..input.d {
        for &j in &unused[i] {
            turn_es[(i + input.d / 2) % input.d].push(j);
        }
    }

    for i in 0..input.d {
        for &j in &turn_es[i] {
            res[j] = i;
        }
    }
    print_ans(res);
}
fn solve3(input: Input) {
    let mut res = vec![0; input.m];
    let mut day_count = vec![0; input.d];
    let group_es = (input.n + input.d - 1) / input.d;
    let mut fl = false;
    for i in 0..input.m {
        let idx = if fl {
            input.es[i].0 / group_es
        } else {
            input.es[i].1 / group_es
        };
        fl = !fl;
        if day_count[idx] != input.k {
            res[i] = idx;
            day_count[idx] += 1;
        }
    }
    print_ans(res);
}

fn solve(input: Input) {
    let mut rng = Xorshift::new();
    let mut cs = State::new(&input);
    let mut loop_cnt = 0;
    let start_time = get_time();
    loop {
        let now_time = get_time();
        if now_time > 5.9 {
            break;
        }
        loop_cnt += 1;
        let edge_idx = rng.rand(input.m);
        let mut next_day = rng.rand(input.d);
        while next_day != cs.res[edge_idx] && cs.day_count[next_day] == input.k {
            next_day = rng.rand(input.d);
        }
        let cur_score = cs.try_apply(&input, edge_idx, next_day);
        if cur_score.0 > cs.cur_score {
            cs.apply(edge_idx, next_day, cur_score);
        }

        // let mut edge_idx2 = rng.rand(input.m);
        // while cs.res[edge_idx] == cs.res[edge_idx2] {
        //     edge_idx2 = rng.rand(input.m);
        // }
        // let cur_score = cs.try_swap_edge(&input, edge_idx, edge_idx2);
        // if cur_score.0 > cs.cur_score {
        //     cs.swap_edge(edge_idx, edge_idx2,cur_score);
        // }
    }
    eprintln!("loops: {}", loop_cnt);
    print_ans(cs.res);
}

fn score(ans: &Vec<usize>, dist: &Vec<Vec<u32>>, input: &Input) -> u64 {
    let mut turn_es = vec![vec![]; input.d];
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

#[derive(Debug)]
struct State {
    day_count: Vec<usize>,
    dist: Vec<Vec<u32>>,
    // ??????????????????????????????
    cur_score: u64,
    res: Vec<usize>,
    idx: Vec<usize>,
    turn_scores: Vec<u64>,
    turn_es: Vec<Vec<usize>>,
}

impl State {
    fn new(input: &Input) -> Self {
        let dist = dijkstra(input.n, &input.es);
        let mut res = vec![0; input.m];
        let mut day_count = vec![0; input.d];
        let rem = input.m % input.d;
        let sho = input.m / input.d;
        let mut now = 0;
        for i in 0..input.d {
            for _ in 0..sho {
                res[now] = i;
                now += 1;
                day_count[i] += 1;
            }
            if i < rem {
                res[now] = i;
                now += 1;
                day_count[i] += 1;
            }
        }
        let mut turn_es = vec![vec![]; input.d];
        for i in 0..input.m {
            for j in 0..input.d {
                if j == res[i] {
                    turn_es[j].push(i);
                }
            }
        }
        let mut idx = vec![0; input.m];
        for i in 0..input.d {
            for j in 0..turn_es[i].len() {
                idx[turn_es[i][j]] = j;
            }
        }
        let mut cur_score = 0;
        let mut turn_scores = vec![0; input.d];
        for i in 0..input.d {
            for j in 0..turn_es[i].len() {
                for k in j + 1..turn_es[i].len() {
                    let d = min!(
                        dist[input.es[turn_es[i][j]].0][input.es[turn_es[i][k]].0],
                        dist[input.es[turn_es[i][j]].1][input.es[turn_es[i][k]].0],
                        dist[input.es[turn_es[i][j]].0][input.es[turn_es[i][k]].1],
                        dist[input.es[turn_es[i][j]].1][input.es[turn_es[i][k]].1]
                    ) as u64;
                    turn_scores[i] += d * d;
                }
            }
            // let sz = turn_es[i].len();
            // let diff = sz.abs_diff(input.m / input.d);
            // let coef = (diff + 1) * (diff + 1);
            // cur_score += turn_scores[i] * coef as u64;
            cur_score += turn_scores[i];
        }
        Self {
            day_count,
            dist,
            cur_score,
            idx,
            res,
            turn_scores,
            turn_es,
        }
    }
    fn try_apply(&self, input: &Input, edge_idx: usize, next_day: usize) -> (u64, u64, u64) {
        let mut now = self.cur_score;
        // ???????????????????????????
        let mut next1 = self.turn_scores[self.res[edge_idx]];
        now -= next1;

        // let sz = self.turn_es[self.res[edge_idx]].len();
        // let diff = sz.abs_diff(input.m / input.d);
        // let coef = (diff + 1) * (diff + 1);
        // next1 /= coef as u64;

        for &i in &self.turn_es[self.res[edge_idx]] {
            // next1 -= self.edge_dist(&input, i, edge_idx) as u64;
            next1 -= self.edge_dist2(&input, i, edge_idx);
        }

        // let sz = self.turn_es[self.res[edge_idx]].len() - 1;
        // let diff = sz.abs_diff(input.m / input.d);
        // let coef = (diff + 1) * (diff + 1);
        // next1 *= coef as u64;

        now += next1;
        // ??????????????????????????????
        let mut next2 = self.turn_scores[next_day];
        now -= next2;

        // let sz = self.turn_es[next_day].len();
        // let diff = sz.abs_diff(input.m / input.d);
        // let coef = (diff + 1) * (diff + 1);
        // next2 /= coef as u64;

        for &i in &self.turn_es[next_day] {
            // next2 += self.edge_dist(&input, i, edge_idx) as u64;
            next2 += self.edge_dist2(&input, i, edge_idx);
        }

        // let sz = self.turn_es[next_day].len() + 1;
        // let diff = sz.abs_diff(input.m / input.d);
        // let coef = (diff + 1) * (diff + 1);
        // next2 *= coef as u64;

        now += next2;
        (now, next1, next2)
    }
    fn edge_dist(&self, input: &Input, a: usize, b: usize) -> u32 {
        min!(
            self.dist[input.es[a].0][input.es[b].0],
            self.dist[input.es[a].1][input.es[b].0],
            self.dist[input.es[a].0][input.es[b].1],
            self.dist[input.es[a].1][input.es[b].1]
        )
    }
    fn apply(
        &mut self,
        edge_idx: usize,
        next_day: usize,
        (next_score, next1, next2): (u64, u64, u64),
    ) {
        self.cur_score = next_score;
        let prev_day = self.res[edge_idx];
        self.day_count[prev_day] -= 1;
        self.day_count[next_day] += 1;
        self.res[edge_idx] = next_day;
        self.turn_scores[prev_day] = next1;
        self.turn_scores[next_day] = next2;

        let back = self.turn_es[prev_day][self.turn_es[prev_day].len() - 1];
        self.idx[back] = self.idx[edge_idx];
        self.turn_es[prev_day].swap_remove(self.idx[edge_idx]);
        self.idx[edge_idx] = self.turn_es[next_day].len();
        self.turn_es[next_day].push(edge_idx);
    }
    fn edge_dist2(&self, input: &Input, a: usize, b: usize) -> u64 {
        let d = self.edge_dist(input, a, b) as u64;
        d * d
    }
    fn try_swap_edge(&self, input: &Input, edge_idx: usize, edge_idx2: usize) -> (u64, u64, u64) {
        let mut now = self.cur_score;
        let mut next1 = self.turn_scores[self.res[edge_idx]];
        now -= next1;
        // ???1???????????????2?????????
        for &i in &self.turn_es[self.res[edge_idx]] {
            if i == edge_idx {
                continue;
            }
            // next1 -= self.edge_dist(&input, i, edge_idx) as u64;
            // next1 += self.edge_dist(&input, i, edge_idx2) as u64;
            next1 -= self.edge_dist2(&input, i, edge_idx);
            next1 += self.edge_dist2(&input, i, edge_idx2);
        }
        now += next1;

        // ???1???????????????2?????????
        let mut next2 = self.turn_scores[self.res[edge_idx2]];
        now -= next2;
        for &i in &self.turn_es[self.res[edge_idx2]] {
            if i == edge_idx2 {
                continue;
            }
            // next2 += self.edge_dist(&input, i, edge_idx) as u64;
            // next2 -= self.edge_dist(&input, i, edge_idx2) as u64;
            next2 += self.edge_dist2(&input, i, edge_idx);
            next2 -= self.edge_dist2(&input, i, edge_idx2);
        }
        now += next2;

        (now, next1, next2)
    }
    fn swap_edge(&mut self, edge_idx: usize, edge_idx2: usize, scores: (u64, u64, u64)) {
        let next_day1 = self.res[edge_idx2];
        let next_day2 = self.res[edge_idx];
        self.apply(edge_idx2, next_day2, scores);
        self.apply(edge_idx, next_day1, scores);
    }
}

fn dijkstra(n: usize, es: &Vec<(usize, usize, u32)>) -> Vec<Vec<u32>> {
    let mut g = vec![vec![]; n];
    for e in es {
        g[e.0].push((e.1, e.2));
        g[e.1].push((e.0, e.2));
    }
    let mut res = vec![vec![T9; n]; n];
    let mut que = BinaryHeap::new();
    for i in 0..n {
        res[i][i] = 0;
        que.push((Reverse(0), i));
        while let Some((Reverse(d), p)) = que.pop() {
            if res[i][p] < d {
                continue;
            }
            for e in &g[p] {
                if chmin!(res[i][e.0], d + e.1) {
                    que.push((Reverse(res[i][e.0]), e.0));
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
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ????????????????????????????????????????????????????????????get_time??????????????????????????????
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.5
        }
        #[cfg(not(feature = "local"))]
        {
            (ms - STIME)
        }
    }
}

pub mod acl {
    pub struct Dsu {
        n: usize,
        // root node: -1 * component size
        // otherwise: parent
        parent_or_size: Vec<i32>,
    }

    impl Dsu {
        pub fn new(size: usize) -> Self {
            Self {
                n: size,
                parent_or_size: vec![-1; size],
            }
        }

        pub fn merge(&mut self, a: usize, b: usize) -> usize {
            assert!(a < self.n);
            assert!(b < self.n);
            let (mut x, mut y) = (self.leader(a), self.leader(b));
            if x == y {
                return x;
            }
            if -self.parent_or_size[x] < -self.parent_or_size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.parent_or_size[x] += self.parent_or_size[y];
            self.parent_or_size[y] = x as i32;
            x
        }

        pub fn same(&mut self, a: usize, b: usize) -> bool {
            assert!(a < self.n);
            assert!(b < self.n);
            self.leader(a) == self.leader(b)
        }

        pub fn leader(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            if self.parent_or_size[a] < 0 {
                return a;
            }
            self.parent_or_size[a] = self.leader(self.parent_or_size[a] as usize) as i32;
            self.parent_or_size[a] as usize
        }

        pub fn size(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            let x = self.leader(a);
            -self.parent_or_size[x] as usize
        }

        pub fn groups(&mut self) -> Vec<Vec<usize>> {
            let mut leader_buf = vec![0; self.n];
            let mut group_size = vec![0; self.n];
            for i in 0..self.n {
                leader_buf[i] = self.leader(i);
                group_size[leader_buf[i]] += 1;
            }
            let mut result = vec![Vec::new(); self.n];
            for i in 0..self.n {
                result[i].reserve(group_size[i]);
            }
            for i in 0..self.n {
                result[leader_buf[i]].push(i);
            }
            result
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect::<Vec<Vec<usize>>>()
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
