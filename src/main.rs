use proconio::{input,marker::Usize1};

fn main() {
    let input = Input::from_input();
    solve(input);
}

fn solve(input: Input) {
    let mut res = vec![0; input.m];
    let mut cnt = 0;
    let mut now = 0;
    for i in 0..input.m {
        res[i] = now;
        cnt += 1;
        if cnt == input.k {
            cnt = 0;
            now += 1;
        }
    }
    print_ans(res);
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
