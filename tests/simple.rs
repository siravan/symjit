use anyhow::Result;
pub use num_complex::{Complex, ComplexFloat};
use symjit::{Composer, Config, Defuns, Slot, Translator};

fn compile_evaluators() -> Result<symjit::Application> {
    let mut config = Config::default();
    config.set_complex(false);
    config.set_direct(false);

    let mut inner = Translator::new(config);
    inner.set_num_params(2);
    inner.append_mul(&Slot::Out(0), &[Slot::Param(0), Slot::Param(1)], 0)?;
    let f = inner.compile()?.seal()?;

    let mut defuns = Defuns::new();
    defuns.add_applet("f", f);

    let mut config = Config::default();
    config.set_complex(false);
    config.set_direct(false);
    config.set_defuns(defuns);

    let mut outer = Translator::new(config);
    outer.set_num_params(2);

    outer.append_fun(
        &Slot::Temp(0),
        "f",
        &[Slot::Param(0), Slot::Param(1)],
        false,
    )?;

    outer.append_add(&Slot::Out(0), &[Slot::Temp(0), Slot::Param(1)], 0)?;
    Ok(outer.compile()?)
}

#[test]
fn main() {
    let args = [5.0, 2.0];
    let expected = 12.0;
    let ret = compile_evaluators().unwrap().evaluate_single(&args);

    println!("expected : {expected}");
    println!("returned : {ret}");

    assert_eq!(ret, expected);
}
