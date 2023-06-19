use std::cmp::{max, min};
use std::collections::HashMap;
use std::collections::VecDeque;


fn main() {

    println!("TEST1");
    let output = shortest_path_binary_matrix(vec![vec![0,1],vec![1,0]]);
    println!("{:?}", output);

    println!("TEST2");
    let output = shortest_path_binary_matrix(vec![vec![0,0,0],vec![1,1,0],vec![1,1,0]]);
    println!("{:?}", output);

    println!("TEST3");
    let output = shortest_path_binary_matrix(
        vec![
        vec![0,1,1,0,0,0], vec![0,1,0,1,1,0],
        vec![0,1,1,0,1,0], vec![0,0,0,1,1,0],
        vec![1,1,1,1,1,0], vec![1,1,1,1,1,0]
        ]
    );
    println!("{:?}", output);

    println!("TEST4");
    let output = shortest_path_binary_matrix(vec![vec![1,0,0],vec![1,1,0],vec![1,1,0]]);
    println!("{:?}", output);

    println!("TEST5");
    let output = shortest_path_binary_matrix(
        vec![
            vec![0,1,0,0,0,0], vec![0,1,0,1,1,0],
            vec![0,1,1,0,1,0], vec![0,0,0,0,1,0],
            vec![1,1,1,1,1,0], vec![1,1,1,1,1,0]
            ]
        );
    println!("{:?}", output);
// [[0,0,0,0,1],[1,0,0,0,0],[0,1,0,1,0],[0,0,0,1,1],[0,0,0,1,0]]

    println!("TEST5");
    let output = shortest_path_binary_matrix(
        vec![
            vec![0,0,0,0,1],vec![1,0,0,0,0],
            vec![0,1,0,1,0],vec![0,0,0,1,1],
            vec![0,0,0,1,0]
            ]
        );

    println!("{:?}", output);
}


pub fn max_satisfaction(mut satisfaction: Vec<i32>) -> i32 {
        satisfaction.sort();
        satisfaction.reverse();
        let mut result = 0;
        let mut best_dishes: Vec<i32> = Vec::new();
        for dish in satisfaction {
            best_dishes.push(dish);
            let mut e_result = 0;
            best_dishes.sort();
            for (index, &item) in best_dishes.iter().enumerate() {
                e_result += (index as i32 + 1) * item;
                println!("Index is {index}, item is {item}, sum is {e_result}")
            }
            if result < e_result {
                result = e_result
            } else {
                best_dishes.pop();
            }
        }
        result
    }


pub fn shortest_palindrome(s: String) -> String {
    let mut result: Vec<char> = Vec::new();
    let mut adding_symbols = String::new();
    for lit in s.chars() {
        result.push(lit);
        let mut n = result.clone();
        n.reverse();
        if result == n {
            adding_symbols = String::from("");
        } else {
            adding_symbols.push(lit);
        }
    }
    let mut boxr = Vec::from(adding_symbols);
    boxr.reverse();
    let a = String::from_utf8(boxr).unwrap();
    let output = a + s.as_str();

    return output
}


pub fn contains_nearby_almost_duplicate(mut nums: Vec<i32>, mut index_diff: i32, value_diff: i32) -> bool {
    if index_diff == 0 { return false;}
    while index_diff != 0 {
        let mut i = 0;
        let mut j = index_diff as usize;
        index_diff -= 1;
        while j != nums.len() as usize {
            if (nums[i] - nums[j]).abs() <= value_diff { return true;}
            j += 1;
            i += 1;
        }
    }
    false
}


pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut arr: HashMap<i32, i32> = HashMap::new();
    for x in 0..nums.len() {
        let y = target - nums[x];
        if arr.contains_key(&y) {
            return vec![x as i32, *arr.get(&y).unwrap()]
        }
        arr.insert(nums[x], x as i32);
    }
    vec![]
}


pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
    let mut arr:Vec<i32> = nums1.clone();
    for i in nums2 {
        match arr.binary_search(&i) {
            Ok(x) => arr.insert(x, i),
            Err(x) => {arr.insert(x, i)}
        }
    }
    if (arr.len() - 1) % 2 == 0{
        return arr[arr.len() / 2] as f64
    } else {
        return (arr[arr.len() / 2] + arr[(arr.len()-1) / 2]) as f64 / 2.0
    }
}


pub fn running_sum(nums: Vec<i32>) -> Vec<i32> {
    let mut output: Vec<i32> = Vec::new();
    for i in 0..nums.len() {
        output.push(nums[0..=i].iter().sum())
    }
    return output
}


pub fn pivot_index(nums: Vec<i32>) -> i32 {
    let a: i32 = nums[1..nums.len()].iter().sum();
    if a == 0 { return 0}
    let a: i32 = nums[0..nums.len()-1].iter().sum() ;
    if  a == 0 {return (nums.len()-1) as i32}
    for num in 1..nums.len()-1 {
        let a: i32 = nums[0..=num-1].iter().sum();
        let b: i32 = nums[num+1..nums.len()].iter().sum();
        if  a == b { return num as i32}
    }
    return -1
}


pub fn is_isomorphic(s: String, t: String) -> bool {
    let s = Vec::from(s);
    let t = Vec::from(t);
    let mut hash_s: HashMap<u8, u8> = HashMap::new();
    let mut hash_t: HashMap<u8, u8> = HashMap::new();
    for lit in 0..s.len() {
        if hash_s.contains_key(&s[lit]) {
            if hash_s.get(&s[lit]).unwrap() != &t[lit] {return false}
        } else {
            if hash_t.contains_key(&t[lit]) {return false}
            hash_s.insert(s[lit], t[lit]);
            hash_t.insert(t[lit], s[lit]);
        }
    }
    return true
}


pub fn is_subsequence(s: String, t: String) -> bool {
    let t = Vec::from(t);
    let s = Vec::from(s);
    let mut l = 0;
    for x in t {
        if x == s[l] {
            l += 1;
            if l == s.len() { return true}
        }
    }
    return false
}


pub fn length_of_longest_substring(s: String) -> i32 {
    let s = Vec::from(s);
    let mut hash: HashMap<u8, usize> = HashMap::new();
    let mut current_l = 0;
    let mut max_l = 0;
    for x in 0..s.len() {
        let mut arr: Vec<u8> = Vec::new();
        if hash.contains_key(&s[x]) {
            if max_l < current_l {max_l = current_l;}
            current_l = x - hash.get(&s[x]).unwrap();
            let t: &usize =  hash.get((&s[x])).unwrap();
            for key in hash.keys() {
                if hash.get(key).unwrap() <= t {arr.push(*key)}
            }
            for key in &arr {
                let _a = hash.remove(&key);
            }
            hash.insert(s[x], x);
        } else {
            current_l += 1;
            hash.insert(s[x], x);
        }
    }
    if max_l < current_l {max_l = current_l}
    return max_l as i32
}


pub fn convert(mut s: String, mut num_rows: i32) -> String {
    let mut output = String::new();
    if num_rows == 1 {return s}
    let mut flag = 0;
    let mut vertical_step: usize = 0;
    let mut index: usize = 0;
    let mut ffff = 0;
    if num_rows == 2 {
        vertical_step = 1;
    } else {
        vertical_step = 3 + (num_rows as usize-3) * 2;
    }
    while num_rows != 1 {
        if index == 0 {
            let a = s.remove(0);
            output += String::from(a).as_str();
        }
        if index + vertical_step >= s.len() {
            if flag == 1 && index + 1 < s.len() {
                let a = s.remove(index+1);
                output += String::from(a).as_str();
                flag = 0;
            }
            println!("{output}");
            println!("{s}");
            if vertical_step -2 < 0 { vertical_step = 0}
            else {  vertical_step -= 2; }
            ffff += 1;
            num_rows -= 1;
            index = 0;
        } else {
            let a = s.remove(index+vertical_step);
            output +=  String::from(a).as_str();
            index += vertical_step;
            if ffff > 0 { flag = 1; }
        }
    }
    output += s.as_str();
    return output
}


pub fn reverse(mut x: i32) -> i32 {
    if x == -2147483648 { return 0}
    let mut result: i32 = 0;
    let mut is_negative: bool = false;
    if x < 0 {
        x = -x;
        is_negative = true;
    }
    while x != 0 {
        if result > 2147483647 / 10 {return 0}
        result = (result*10) + x%10;
        x /= 10;
    }
    if is_negative { return result*(-1)}
    return result
}


pub fn my_atoi(s: String) -> i32 {
    let s = s.trim_start_matches(|c| c == ' ');
    let (sign, s) = match s.strip_prefix(|c| c == '-') {
        Some(s) => (-1, s),
        None => (1, s.strip_prefix(|c| c == '+').unwrap_or(s)),
    };
    s.chars().map_while(|c| c.to_digit(10)).fold(0, |acc, x| {
        acc.saturating_mul(10).saturating_add(sign * x as i32)
    })
}


pub fn is_palindrome(x: i32) -> bool {
    let mut a = Vec::from(x.to_string());
    a.reverse();
    let b = String::from_utf8(a).unwrap();
    x.to_string() == b
}


pub fn minimize_array_value(mut nums: Vec<i32>) -> i32 {
    for i in 1..nums.len() {
        if nums[i-1] > nums[i] { continue }
        if (nums[i] + nums[i-1]) % 2 == 0 {
            let a = (nums[i]+nums[i-1]) / 2;
            nums[i] = a;
            nums[i-1] = a
        } else {
            let a = (nums[i] + nums[i - 1] + 1) / 2;
            nums[i - 1] = a - 1;
            nums[i] = a;
        }
    }
    let mut m= 0;
    for n in nums {
        if m < n { m = n}
    }
    return m
}


pub fn four_sum(mut nums: Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    nums.sort();
    let mut arr: Vec<Vec<i32>> = vec![Vec::new()];

    return arr
}


pub fn count_good_rectangles(rectangles: Vec<Vec<i32>>) -> i32 {
    let mut square: i32 = 0;
    let mut counter: i32 = 0;
    let mut max_counter: i32 = 0;
    for rectangle in rectangles {
        if rectangle[0] == rectangle[1] {
            if square < rectangle[0] {
                square = rectangle[0];
                counter = 1;
            } else if square == rectangle[0] {
                counter += 1
            }
        }
        else {
            let current_square =  min(rectangle[0], rectangle[1]);
            if square < current_square {
                square = current_square;
                counter = 1;
            } else if square == current_square {
                counter += 1;
            }
        }
    }
    max_counter = max(1, max(max_counter, counter));
    return max_counter
}


pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
    let mut count = 0;
    let mut map = std::collections::HashMap::new();
    for i in 0..nums.len() {
        for j in i + 1..nums.len() {
            let product = nums[i] * nums[j];
            let entry = map.entry(product).or_insert(0);
            *entry += 1;
        }
    }
    for (_, v) in map {
        count += v * (v - 1) / 2 * 8;
    }
    count
}


pub fn find_subsequences(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut index = 2;
    if nums.len() == 1 {return vec![Vec::new()]}
    let mut result: Vec<Vec<i32>> = Vec::new();
    result.push(vec![nums[0], nums[1]]);
    let mut max_num = max(nums[0], nums[1]);
    while nums.len() - index > 0 {
        if nums[index] > max_num {
        }
    }




    result
}


pub fn longest_palindrome(s: String) -> String {
    fn is_palidrone(s: &[u8]) -> bool {
        s.iter().zip(s.iter().rev()).all(|(l, r)| l == r)
    }

    for size in (1..=s.len()).rev() {
        match s.as_bytes()
            .windows(size)
            .find(|substr| is_palidrone(substr)) {
            Some(pal) => return String::from_utf8(pal.to_vec()).unwrap(),
            None => continue,
        }
    }
    String::from("")
}


pub fn longest_common_prefix(input: Vec<String>) -> String {
    input.into_iter().reduce(|acc,cur|{
        acc.chars()
            .zip(cur.chars())
            .take_while(|(a,c)| a== c)
            .map(|(c,_)|c)
            .collect()
    }).unwrap()
}


pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
    let mut output: Vec<Vec<i32>> = Vec::new();
    let mut counter: i32 = 0;
    for z in 0..nums.len() {
        for y in (z+1)..nums.len() {
            for x in (y+1)..nums.len() {
                if nums[x] + nums[y] + nums[z] == 0 {
                    let mut b = vec![nums[x], nums[y], nums[z]];
                    b.sort();
                    if output.contains(&b) == false {
                        counter += 1;
                        output.push(b);
                    }
                }
            }
        }
    }
    return output
}


pub fn is_valid(s: String) -> bool {
    let mut open_brackets: Vec<char> = vec![];
    for p in s.chars(){
        if p == '}' || p == ')' || p == ']'{
            if open_brackets.is_empty(){
                return false;
            }
            let temp_bracket: char = open_brackets[open_brackets.len()-1].clone();
            if p == ']' && temp_bracket != '['{return false}
            if p == '}' && temp_bracket != '{'{return false}
            if p == ')' && temp_bracket != '('{return false}
            open_brackets.pop();
            continue;
        }
        open_brackets.push(p);
    }
    open_brackets.is_empty()
}


pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let mut first = 0;
    let mut last = nums.len() - 1;
    let mut found = false;
    while first < last && found == false && last-first > 1 {
        let midpoint = (first + last) / 2;
        if nums[midpoint] == target {
            found = true;
            return midpoint as i32
        } else {
            if target < nums[midpoint] {
                last = midpoint;
            } else {
                first = midpoint
            }
        }
    }
    if nums[0] == target {
        return 0
    }
    if nums[nums.len()-1] == target {
        return (nums.len()-1) as i32
    }
    -1
}


pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let mut hash: HashMap<i32, usize> = HashMap::new();
    let mut hash_length = 0;
    for i in 0..nums.len() {
        if hash.contains_key(&nums[i]) == false {
            hash_length += 1;
        }
        hash.insert(nums[i], i);
    }
    return hash_length
}

pub fn max_vowels(s: String, k: i32) -> i32 {
    fn is_vowel(ch: u8) -> i32 {
        match ch {
            105 | 111 | 97 | 117 | 101 => 1,
            _ => 0,
        }
    }
    let mut res = 0;
    let mut iter = s.as_bytes().windows(k as usize);
    loop {
        match iter.next() {
            Some(arr) => {
                res = max(res, arr.into_iter().fold(0, |acc, x| acc+is_vowel(*x)));
                if res == k {break}
            },
            None => break
        }
    }
    return res
}




pub fn shortest_path_binary_matrix(grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    if grid[0][0] == 1 || grid[n - 1][n - 1] == 1 {
        return -1;
    }
    
    let mut queue = VecDeque::new();
    let mut visited = vec![vec![false; n]; n];
    
    queue.push_back((0, 0, 1));
    visited[0][0] = true;
    
    while let Some((row, col, length)) = queue.pop_front() {
        if row == n - 1 && col == n - 1 {
            return length;
        }
        
        for (dr, dc) in &[
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ] {
            let new_row = row as i32 + dr;
            let new_col = col as i32 + dc;
            
            if new_row >= 0 && new_row < n as i32 && new_col >= 0 && new_col < n as i32 {
                let new_row = new_row as usize;
                let new_col = new_col as usize;
                
                if grid[new_row][new_col] == 0 && !visited[new_row][new_col] {
                    queue.push_back((new_row, new_col, length + 1));
                    visited[new_row][new_col] = true;
                }
            }
        }
    }
    
    -1
}