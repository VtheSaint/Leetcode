use std::cmp::{max, min, self};
use std::collections::{HashMap, HashSet};
use std::collections::VecDeque;

fn main() {
    let result = is_palindrome("A man, a plan, a canal: Panama".to_string());
    println!("{}", result);
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


pub fn is_palindrome_(x: i32) -> bool {
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


pub fn maximum_detonation(mut bombs: Vec<Vec<i32>>) -> i32 {

    // Find first neighbours 
    fn get_bombs(bombs:&mut Vec<Vec<i32>>, bi: usize) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();
        for i in 0..bombs.len() {
            if i == bi {
                result.push(bi);
                continue;
            }
            if ((bombs[i][0] - bombs[bi][0]) as i64).pow(2) + ((bombs[i][1] - bombs[bi][1]) as i64).pow(2) <= (bombs[bi][2] as i64).pow(2) {
                result.push(i);
            }
        }
        
        return result
    }


    let mut bomb_map: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut output = 0;
    // Go through all bombs 
    for cbi in 0..bombs.len() {

        // Check the Hash 
        if bomb_map.contains_key(&cbi) { continue }

        let mut result: Vec<usize> = Vec::new();
        result.push(cbi);
        result.extend(get_bombs(&mut bombs, cbi));
        result.sort();
        result.dedup();
        let mut is_done = false;
        loop {
            // add new neighbours 
            is_done = true;
            for sub_n in result.clone() {
                if bomb_map.contains_key(&sub_n) {
                    continue;
                } else {
                    is_done = false;
                    let res = get_bombs(&mut bombs, sub_n);
                    bomb_map.insert(sub_n, res.clone());
                    result.extend(res);
                    result.sort();
                    result.dedup();
                }
            }
            // check 
            if is_done == true {break;}
        }
        output = max(output, result.len());
    }
    output as i32 
}





pub fn merge(nums1: &mut Vec<i32>, m: i32, nums2: &mut Vec<i32>, n: i32) {
    let  (mut m, mut n) = (m as usize, n as usize);
    while n > 0 {
        if m > 0 && nums1[m - 1] > nums2[n - 1] {
            nums1[m + n - 1] = nums1[m - 1];
            m -= 1;
        } else {
            nums1[m + n - 1] = nums2[n - 1];
            n -= 1;
        }
    }
}


pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
    let mut result = 0;
    for i in 0..nums.len() {
        //
        if nums[i] != val {
            nums[result] = nums[i];
            result += 1;
        }
    }
    result as i32
}


pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let mut hash: HashMap<i32, usize> = HashMap::new();
    let mut hash_length = 0;
    let mut i: usize = 0;
    while i < nums.len() {
        if hash.contains_key(&nums[i]) == false {
            hash_length += 1;
            hash.insert(nums[i], i);
        } else {
            nums.remove(i);
            i -= 1;
        }
        i += 1
        
    }
    return hash_length
}

// Version II
pub fn remove_duplicates_v2(nums: &mut Vec<i32>) -> i32 {
    let mut last = (0, -1);

    nums.retain(|num|{
        let result = last.0 == last.1 && last.1 == *num;

        last.0 = last.1;
        last.1 = *num;
        !result
    });

    nums.len() as i32
}


pub fn majority_element(nums: Vec<i32>) -> i32 {
    let mut hash: HashMap<i32, i32> = HashMap::new();
    let result = 0;
    let l = nums.len() as i32;
    for num in nums {
        if hash.contains_key(&num) {
            let count = hash.get(&num).unwrap().clone();
            hash.remove(&num);
            hash.insert(num, count + 1);
        } else {
            hash.insert(num, 1);
        }
    }
    for num in hash.keys().into_iter() {
        if *hash.get(&num).unwrap() > l / 2 {
            return *num
        }
    }
    -1
}


pub fn rotate(nums: &mut Vec<i32>, k: i32) {
    for i in 0..k {
        let x = nums.pop().unwrap();
        nums.insert(0, x);
    }       
}


pub fn max_profit(prices: Vec<i32>) -> i32 {
    let mut result = 0;
    let mut best_buy_price = i32::MAX;
    let mut best_sell_price = 0;
    for price in prices {
        if price < best_buy_price {
            best_buy_price = price;
            best_sell_price = 0;
        }
        else if price > best_sell_price {
            best_sell_price = price;
            let diff = best_sell_price - best_buy_price;
            if diff > result {
                result = diff;
            }
        }
    }
    result
}


pub fn max_profit_v2(prices: Vec<i32>) -> i32 {
    prices.windows(2).fold(0, |acc, w| acc + (w[1] - w[0]).max(0))
}

pub fn can_jump(nums: Vec<i32>) -> bool {
    let mut max_val: i32 = 0;

    for (idx, val) in nums.iter().enumerate() {
        println!("{:?}",  (idx, val));
        if idx as i32 > max_val {
            return false;
        }
        max_val = std::cmp::max(max_val, (idx as i32 + *val as i32) as i32);
        println!("MAX_val={:?}", max_val);
    }
    return true
}

// [3,2,1,0,4]


pub fn minimum_deletions(nums: Vec<i32>) -> i32 {
    let mut min_num = std::i32::MAX;
    let mut min_index = nums.len();
    let mut max_num = std::i32::MIN;
    let mut max_index = nums.len();
    for num in 0..nums.len() {
        if nums[num] < min_num { 
            min_num = nums[num];
            min_index = num
        }
        if nums[num] > max_num { 
            max_num = nums[num];
            max_index = num;
        }
    }

    // min and max are not inverted
    let mut result = max(min_index, max_index) + 1;

    // min inverted max not 
    min_index = nums.len() - 1 - min_index;
    result = min(result, max_index + min_index + 2);

    // min and max inverted
    max_index = nums.len() - 1 - max_index;
    result = min(result, max(max_index, min_index) + 1);

    // max inverted min not 
    min_index = nums.len() - 1 - min_index;
    result = min(result, max_index + min_index + 2);


    result as i32 
}


pub fn find_all_people(n: i32, mut meetings: Vec<Vec<i32>>, first_person: i32) -> Vec<i32> {
    
    // HashMap<person,(time, knows_secret)>
    let mut persons: HashMap<i32,bool> = HashMap::new();
    for i in 0..n {
        persons.insert(i, false);
    }

    // This people knows the secret at 0 second
    persons.insert(0, true);
    persons.insert(first_person, true);
    
    // Sorting meetings by time 
    meetings.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
    let mut current_time = 0;
    let mut meeting_pool: Vec<(i32, i32)> = Vec::new();
    for m_index in 0..meetings.len() {
        // if we have meetings in the same time 
        if current_time != meetings[m_index][2] {
            // we are in new pool 
            // check the last pool 
            for _ in 0..2 {
                for meeting in &meeting_pool {
                    if persons.get(&meeting.0).unwrap() == &true || persons.get(&meeting.1).unwrap() == &true {
                        persons.insert(meeting.1,  true);
                        persons.insert(meeting.0, true);   
                    }
                }
            }
            // create a new one 
            current_time = meetings[m_index][2];
            meeting_pool = vec![(meetings[m_index][0], meetings[m_index][1])];
        } 
        else { 
            meeting_pool.push((meetings[m_index][0], meetings[m_index][1]))
        }

    }
    for _ in 0..2 {
        for meeting in &meeting_pool {
            if persons.get(&meeting.0).unwrap() == &true || persons.get(&meeting.1).unwrap() == &true {
                persons.insert(meeting.1,  true);
                persons.insert(meeting.0, true);   
            }
        }
    }
    let mut result = persons
        .iter()
        .filter(|person| person.1 == &true)
        .map(|person| person.0.clone())
        .collect::<Vec<i32>>();
    

    result.sort();
    result
}


pub fn minimum_total(mut triangle: Vec<Vec<i32>>) -> i32 {
    if triangle.len() == 1 { return triangle[0][0]}
    while triangle.len() != 2 {
        let cur_row = triangle.len()-2;
        for i in 0..triangle[cur_row].len() {
            triangle[cur_row][i] = min(triangle[cur_row+1][i] + triangle[cur_row][i], triangle[cur_row+1][i+1] + triangle[cur_row][i]);
        }
        triangle.pop();
    }
    triangle[0][0] = min(triangle[0][0] + triangle[1][0], triangle[0][0] + triangle[1][1]);
    return triangle[0][0]
}


pub fn min_path_sum(mut grid: Vec<Vec<i32>>) -> i32 {
    for y in 0..grid.len() {
        for x in 0..grid[0].len() {
            if x == 0 && y == 0 {
                continue;
            }
            else if x == 0 {
                grid[y][x] += grid[y-1][x];
            }
            else if y == 0 {
                grid[y][x] += grid[y][x-1];
            }
            else {
                grid[y][x] = min(grid[y][x] + grid[y][x-1], grid[y][x] + grid[y-1][x]);
            }
        }
    }
    return grid[grid.len()-1][grid[0].len()-1]

}


pub fn climb_stairs(n: i32) -> i32 {
    if n <= 2 {
        return n;
    }
    
    let mut dp = vec![0; n as usize + 1];
    dp[1] = 1;
    dp[2] = 2;
    
    for i in 3..=n as usize {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    
    dp[n as usize]
}


use rand::seq::SliceRandom;

struct RandomizedSet {
    set: HashSet<i32>,
    v: Vec<i32>

}


/** 
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl RandomizedSet {

    fn new() -> Self {
        Self { 
            set: HashSet::new(),
            v: Vec::new()
        }
    }
    
    fn insert(&mut self, val: i32) -> bool {
        if self.set.contains(&val) {false} else {self.set.insert(val); self.v.push(val); true}
    }
    
    fn remove(&mut self, val: i32) -> bool {
        if self.set.contains(&val) {false} else {self.set.remove(&val); self.remove(val); true}
    }
    
    fn get_random(&self) -> i32 {
        *self.v.choose(&mut rand::thread_rng()).unwrap()
    }
}

pub fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
    let mut res = vec![];
    let mut prefix = 1;

    for i in 0..nums.len() {
        res.push(prefix);
        prefix *= nums[i];
    }

    let mut postfix = 1;
    for i in (0..nums.len()).rev() {
        res[i] *= postfix;
        postfix *= nums[i];
    }
    
    res
}


pub fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
    for x in 0..gas.len() {
        let mut current = gas[x];
        for y in 1..=gas.len() {
            let current_gas = (x + y - 1) % gas.len();
            let next_gas = (x + y)%gas.len();
            current -= cost[current_gas];
            if current < 0 {
                current = -1;
                break;
            }
            current += gas[next_gas];
        }
        if current < 0 {
            current = 0;
        } else if current > 0 {
            return x as i32;
        }
    }
    return -1;
}



pub fn unique_paths_with_obstacles(mut obstacle_grid: Vec<Vec<i32>>) -> i32 {

    for y in 0..obstacle_grid.len() { 
        for x in 0..obstacle_grid[0].len() {
            if obstacle_grid[y][x] == 1 { obstacle_grid[y][x] = -1}
        }
    }
    if obstacle_grid[0][0] == -1 {return 0} else { obstacle_grid[0][0] = 1};
    for y in 0..obstacle_grid.len() {
        for x in 0..obstacle_grid[0].len() {
            if obstacle_grid[y][x] == -1 { 
                continue;
            }
            if y == obstacle_grid.len() - 1  && x == obstacle_grid[0].len() - 1 {
                continue;
            }
            else if y == obstacle_grid.len() - 1 {
                if obstacle_grid[y][x + 1] != -1 {
                    obstacle_grid[y][x+1] += obstacle_grid[y][x];
                }
                
            }
            else if x == obstacle_grid[0].len() - 1 {
                if obstacle_grid[y+1][x] != -1 {
                    obstacle_grid[y+1][x] += obstacle_grid[y][x];
                }
            }
            else { 
                if obstacle_grid[y+1][x] != -1 {
                    obstacle_grid[y+1][x] += obstacle_grid[y][x];
                }
                if obstacle_grid[y][x+1] != -1 {
                    obstacle_grid[y][x+1] += obstacle_grid[y][x];
                }
            }
        }
    }
    let result = *obstacle_grid.last().unwrap().last().unwrap();
    if result == -1 {return 0} else {return result}

}


pub fn maximal_square(matrix: Vec<Vec<char>>) -> i32 {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut dp = vec![vec![0; cols + 1]; rows + 1];
    let mut max_len: i32 = 0;

    for row in 1..=rows {
        for col in 1..=cols {
            if matrix[row - 1][col - 1] == '1' {
                dp[row][col] = 1 + cmp::min(
                    cmp::min(dp[row - 1][col - 1], dp[row - 1][col]),
                    dp[row][col - 1],
                );
                max_len = cmp::max(max_len, dp[row][col] as i32);
            }
        }
    }

    max_len * max_len
}



pub fn is_palindrome(s: String) -> bool {
    let ss = s
        .as_bytes()
        .iter()
        .filter(|x| x.is_ascii_alphanumeric())
        .map(|x| x.to_ascii_lowercase())
        .collect::<Vec<u8>>();
    let mut s2 = ss.clone();
    s2.reverse();
    return ss == s2
}